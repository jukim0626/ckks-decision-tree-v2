"""epoch_worker.py(local_loss)를 forward_backward_update_N_packed용으로 바꾼 버전 -
packed/epoch_worker_packed.py와 완전히 같은 목적/구조. setup/finalize는 params[] 포맷이
packing과 무관해서 local_loss의 기존 setup_worker.py/finalize_worker.py를 그대로 재사용
(packed 전용 setup 파일 불필요) - blocked_features/sample_mask_blocked/block_masks/
block_size는 dataset/sample_mask의 순수 함수(rotation만, bootstrap 없음)라 이 프로세스
시작 시 매번 새로 계산한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data.serialization import load_context, read_dataset  # noqa: E402
from core.ckks_engine import create_bootstrap_engine  # noqa: E402
from models.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    assert_layout_fits,
    broadcast_full_to_blocks,
    build_block_masks,
    compute_block_size,
    pack_dataset_features_blocked,
)
from models.gradient_soft_tree.local_loss.tree_ops_packed import forward_backward_update_N_packed  # noqa: E402
from models.gradient_soft_tree.params import (  # noqa: E402
    load_alpha,
    load_local_logits,
    load_threshold,
    save_alpha,
    save_local_logits,
    save_threshold,
)


def main() -> None:
    session_dir = Path(sys.argv[1])
    config = json.loads((session_dir / "config.json").read_text())
    n_features = config["n_features"]
    depth = config["depth"]
    n_internal = (1 << depth) - 1

    engine = create_bootstrap_engine(
        mode=config["mode"], device_id=config["device_id"], level_preset=config.get("level_preset")
    )
    ctx = load_context(engine, session_dir / "keys", mode=config["mode"], device_id=config["device_id"])
    dataset = read_dataset(
        ctx,
        session_dir / "dataset",
        n_features=n_features,
        n_classes=config["n_classes"],
        n_samples=config["n_samples"],
    )
    sample_mask = engine.read_ciphertext(session_dir / "sample_mask.ct")

    block_size = compute_block_size(dataset.n_samples)
    assert_layout_fits(n_features, block_size, engine.slot_count)
    block_masks = build_block_masks(n_features, block_size, engine.slot_count)
    blocked_features = pack_dataset_features_blocked(ctx, dataset.enc_features, block_size)
    sample_mask_blocked = broadcast_full_to_blocks(ctx, sample_mask, n_features, block_size)

    params_dir = session_dir / "params"
    params = {
        "alpha": load_alpha(engine, params_dir, n_internal),
        "threshold": load_threshold(engine, params_dir, n_internal, n_features),
        "local_logits": load_local_logits(engine, params_dir, depth),
    }

    new_params = forward_backward_update_N_packed(
        ctx, dataset, params, sample_mask, blocked_features, sample_mask_blocked,
        block_masks, block_size, n_features, config["n_classes"], depth, lr=config["lr"],
    )

    save_alpha(engine, params_dir, new_params["alpha"])
    save_threshold(engine, params_dir, new_params["threshold"])
    save_local_logits(engine, params_dir, new_params["local_logits"])


if __name__ == "__main__":
    main()
