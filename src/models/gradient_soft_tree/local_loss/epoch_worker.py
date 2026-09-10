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
        "alpha": [engine.read_ciphertext(params_dir / f"alpha_{i}.ct") for i in range(n_internal)],
        "threshold": [
            [engine.read_ciphertext(params_dir / f"threshold_{i}_{j}.ct") for j in range(n_features)]
            for i in range(n_internal)
        ],
        "local_logits": [
            [engine.read_ciphertext(params_dir / f"local_{level}_{k}.ct") for k in range(1 << (level + 1))]
            for level in range(depth)
        ],
    }

    new_params = forward_backward_update_N(
        ctx, dataset, params, sample_mask, n_features, config["n_classes"], depth, lr=config["lr"]
    )

    for i in range(n_internal):
        engine.write_ciphertext(new_params["alpha"][i], params_dir / f"alpha_{i}.ct")
        for j in range(n_features):
            engine.write_ciphertext(new_params["threshold"][i][j], params_dir / f"threshold_{i}_{j}.ct")
    for level in range(depth):
        for k in range(1 << (level + 1)):
            engine.write_ciphertext(new_params["local_logits"][level][k], params_dir / f"local_{level}_{k}.ct")


if __name__ == "__main__":
    main()
