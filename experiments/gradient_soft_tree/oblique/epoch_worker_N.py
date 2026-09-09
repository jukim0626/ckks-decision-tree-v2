"""epoch_worker_N.py(axis-aligned)와 완전히 같은 구조 - w/b ciphertext를 읽고 쓰는 것만 다름."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from closed_form_mgi.io_utils import load_context, read_dataset  # noqa: E402
from closed_form_mgi.primitives import create_bootstrap_engine  # noqa: E402
from experiments.gradient_soft_tree.oblique.depthN_ckks import forward_backward_update_N  # noqa: E402


def main() -> None:
    session_dir = Path(sys.argv[1])
    # 2026-09-09 롤백 메커니즘 추가: 위험 감지 후 재시도할 때 이 epoch만 lr을 낮춰서
    # 다시 돌릴 수 있도록 선택적 lr override를 받는다. 생략(또는 "none")하면 기존과
    # 동일하게 config["lr"] 사용 - 기본 동작은 100% 그대로 하위호환.
    lr_override = None
    if len(sys.argv) > 2 and sys.argv[2] != "none":
        lr_override = float(sys.argv[2])
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

    params_dir = session_dir / "params"
    params = {
        "w": [
            [engine.read_ciphertext(params_dir / f"w_{i}_{j}.ct") for j in range(n_features)]
            for i in range(n_internal)
        ],
        "b": [engine.read_ciphertext(params_dir / f"b_{i}.ct") for i in range(n_internal)],
        "leaf_logits": [engine.read_ciphertext(params_dir / f"leaf_{l}.ct") for l in range(n_leaves)],
    }

    effective_lr = lr_override if lr_override is not None else config["lr"]
    new_params = forward_backward_update_N(
        ctx, dataset, params, sample_mask, n_features, config["n_classes"], depth, lr=effective_lr
    )

    for i in range(n_internal):
        for j in range(n_features):
            engine.write_ciphertext(new_params["w"][i][j], params_dir / f"w_{i}_{j}.ct")
        engine.write_ciphertext(new_params["b"][i], params_dir / f"b_{i}.ct")
    for l in range(n_leaves):
        engine.write_ciphertext(new_params["leaf_logits"][l], params_dir / f"leaf_{l}.ct")


if __name__ == "__main__":
    main()
