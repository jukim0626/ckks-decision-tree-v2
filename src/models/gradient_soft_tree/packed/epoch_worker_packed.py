"""epoch_worker_N.py를 forward_backward_update_N_packed용으로 바꾼 버전. setup은
baseline과 완전히 동일하므로 `setup_worker_N.py`를 그대로 재사용한다(packed 전용 setup
파일이 따로 필요 없음 - 데이터셋/초기 파라미터 암호화 로직은 packing과 무관). finalize도
`params[]` 포맷이 baseline과 동일해서 `finalize_worker_N.py`를 그대로 재사용한다.

이 파일만 packed 전용인 이유: `forward_backward_update_N_packed`를 부르려면
blocked_features/sample_mask_blocked/block_masks/block_size가 필요한데, 전부 (1) dataset/
sample_mask(이미 session_dir에 저장돼 setup_worker_N.py가 만들어둠)의 순수 함수이고 (2)
계산 비용이 싸므로(rotation만, bootstrap 없음) 매 epoch_worker_packed 프로세스 시작 시
새로 계산한다 - 별도로 직렬화/로드할 필요가 없다."""

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
from models.gradient_soft_tree.packed.tree_ops_packed import forward_backward_update_N_packed  # noqa: E402


def main() -> None:
    session_dir = Path(sys.argv[1])
    config = json.loads((session_dir / "config.json").read_text())
    n_features = config["n_features"]
    depth = config["depth"]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth

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
        "alpha": [engine.read_ciphertext(params_dir / f"alpha_{i}.ct") for i in range(n_internal)],
        "threshold": [
            [engine.read_ciphertext(params_dir / f"threshold_{i}_{j}.ct") for j in range(n_features)]
            for i in range(n_internal)
        ],
        "leaf_logits": [engine.read_ciphertext(params_dir / f"leaf_{l}.ct") for l in range(n_leaves)],
    }

    new_params = forward_backward_update_N_packed(
        ctx, dataset, params, sample_mask, blocked_features, sample_mask_blocked,
        block_masks, block_size, n_features, config["n_classes"], depth, lr=config["lr"],
    )

    for i in range(n_internal):
        engine.write_ciphertext(new_params["alpha"][i], params_dir / f"alpha_{i}.ct")
        for j in range(n_features):
            engine.write_ciphertext(new_params["threshold"][i][j], params_dir / f"threshold_{i}_{j}.ct")
    for l in range(n_leaves):
        engine.write_ciphertext(new_params["leaf_logits"][l], params_dir / f"leaf_{l}.ct")


if __name__ == "__main__":
    main()
