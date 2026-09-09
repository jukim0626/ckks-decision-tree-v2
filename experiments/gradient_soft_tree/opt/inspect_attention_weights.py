"""Q4 답변용: 학습 끝난 session_dir에서 각 internal node의 feature attention weight를
decrypt해서 출력 (attention_softmax=False면 raw a_ij 자체가 곧 "weight" - softmax 없이도
feature selection이 sparse/near one-hot로 수렴하는지 육안 확인용).

python -m experiments.gradient_soft_tree.opt.inspect_attention_weights <session_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from closed_form_mgi.io_utils import load_context  # noqa: E402
from closed_form_mgi.primitives import create_bootstrap_engine  # noqa: E402
from experiments.gradient_soft_tree.depth1_reference import softmax  # noqa: E402


def main() -> None:
    session_dir = Path(sys.argv[1])
    config = json.loads((session_dir / "config.json").read_text())
    exp = json.loads((session_dir / "experiment_config.json").read_text())
    n_features = config["n_features"]
    depth = config["depth"]
    n_internal = (1 << depth) - 1

    engine = create_bootstrap_engine(mode=config["mode"], device_id=config["device_id"], level_preset=config.get("level_preset"))
    ctx = load_context(engine, session_dir / "keys", mode=config["mode"], device_id=config["device_id"])

    params_dir = session_dir / "params"
    print(f"preset={exp['name']} attention_softmax={exp['attention_softmax']} lambda_sum={exp['lambda_sum']} lambda_binary={exp['lambda_binary']}")
    for i in range(n_internal):
        raw = np.real(ctx.engine.decrypt(engine.read_ciphertext(params_dir / f"alpha_{i}.ct"), ctx.sk))[:n_features]
        weight = softmax(raw) if exp["attention_softmax"] else raw
        print(f"node{i}: raw_alpha={np.round(raw, 4)}  weight={np.round(weight, 4)}  sum={weight.sum():.4f}")


if __name__ == "__main__":
    main()
