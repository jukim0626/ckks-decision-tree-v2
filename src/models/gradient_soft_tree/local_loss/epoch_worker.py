"""local_loss forward_backward_update_N의 한 epoch을 별도 프로세스로 실행
(baseline/epoch_worker.py와 동일한 목적/구조 - GPU 메모리 회수를 위한 프로세스 격리).

session_dir/params/alpha_{i}.ct, threshold_{i}_{j}.ct, local_{level}_{k}.ct를 읽어서
한 epoch 갱신한 뒤 같은 자리에 덮어쓴다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.serialization import load_context, read_dataset  # noqa: E402
from core.ckks_engine import create_bootstrap_engine  # noqa: E402
from models.gradient_soft_tree.local_loss.tree_ops import forward_backward_update_N  # noqa: E402
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

    params_dir = session_dir / "params"
    params = {
        "alpha": load_alpha(engine, params_dir, n_internal),
        "threshold": load_threshold(engine, params_dir, n_internal, n_features),
        "local_logits": load_local_logits(engine, params_dir, depth),
    }

    new_params = forward_backward_update_N(
        ctx, dataset, params, sample_mask, n_features, config["n_classes"], depth, lr=config["lr"]
    )

    save_alpha(engine, params_dir, new_params["alpha"])
    save_threshold(engine, params_dir, new_params["threshold"])
    save_local_logits(engine, params_dir, new_params["local_logits"])


if __name__ == "__main__":
    main()
