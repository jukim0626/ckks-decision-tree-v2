"""매 epoch 끝난 뒤, 실제 CKKS 파라미터(w,b)를 decrypt해서 진짜 z=w.x+b 범위가 안전구간
([-2,2], sigmoid_approx_coeffs.py의 다항식 근사 유효구간)을 얼마나 벗어났는지 관찰하는
전용 워커. **자동으로 뭘 고치진 않고 관찰만 한다** - 2026-09-07 밤 실험에서 "plaintext
trajectory는 안전한데 실제 CKKS는 발산"하는 사례가 반복돼서, 언제/어떻게 실제 경로가
plaintext에서 벗어나 위험구간에 들어가는지 데이터를 먼저 모으기 위한 진단 도구.

finalize_worker_N.py와 같은 지위(검증/평가 목적, protocol 일부 아님) - GPU를 오래 안 잡도록
setup/epoch worker와 마찬가지로 별도 프로세스로 분리."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from client_assisted.dataset import load_scaled_dataset_subset  # noqa: E402
from closed_form_mgi.io_utils import load_context  # noqa: E402
from closed_form_mgi.primitives import create_bootstrap_engine  # noqa: E402
from experiments.gradient_soft_tree.oblique.depthN_ckks import decrypt_params_N  # noqa: E402


def main() -> None:
    session_dir = Path(sys.argv[1])
    epoch = int(sys.argv[2])
    config = json.loads((session_dir / "config.json").read_text())
    n_features = config["n_features"]
    n_classes = config["n_classes"]
    depth = config["depth"]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth

    engine = create_bootstrap_engine(
        mode=config["mode"], device_id=config["device_id"], level_preset=config.get("level_preset")
    )
    ctx = load_context(engine, session_dir / "keys", mode=config["mode"], device_id=config["device_id"])

    params_dir = session_dir / "params"
    final_params = {
        "w": [
            [engine.read_ciphertext(params_dir / f"w_{i}_{j}.ct") for j in range(n_features)]
            for i in range(n_internal)
        ],
        "b": [engine.read_ciphertext(params_dir / f"b_{i}.ct") for i in range(n_internal)],
        "leaf_logits": [engine.read_ciphertext(params_dir / f"leaf_{l}.ct") for l in range(n_leaves)],
    }
    decoded = decrypt_params_N(ctx, final_params, n_features, n_classes, depth)

    X_train, _, _, _, _ = load_scaled_dataset_subset(config["dataset_name"], max_train=config.get("max_train"))

    per_node_max_z = []
    for i in range(n_internal):
        z = X_train @ decoded["w"][i] + decoded["b"][i]
        per_node_max_z.append(float(np.abs(z).max()))

    result = {
        "epoch": epoch,
        "max_z": max(per_node_max_z) if per_node_max_z else 0.0,
        "per_node_max_z": per_node_max_z,
        "max_w_abs": float(np.abs(decoded["w"]).max()),
        "max_b_abs": float(np.abs(decoded["b"]).max()),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
