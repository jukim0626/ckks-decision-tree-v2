"""finalize_worker_N.py(axis-aligned)와 완전히 같은 구조 - decrypt/plaintext reference/
predict를 전부 oblique 버전으로 바꿈."""

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
from experiments.gradient_soft_tree.oblique.depthN_reference import predict as plaintext_predict, train_depthN_oblique  # noqa: E402


def main() -> None:
    session_dir = Path(sys.argv[1])
    config = json.loads((session_dir / "config.json").read_text())
    n_features = config["n_features"]
    n_classes = config["n_classes"]
    depth = config["depth"]
    dataset_name = config["dataset_name"]
    seed = config["seed"]
    lr = config["lr"]
    n_epochs = int(sys.argv[2])
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

    X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(
        dataset_name, max_train=config.get("max_train")
    )
    ref_final = train_depthN_oblique(X_train, np.eye(n_classes)[y_train], depth=depth, lr=lr, epochs=n_epochs, seed=seed)
    max_err = max(
        np.abs(decoded["w"] - ref_final["w"]).max(),
        np.abs(decoded["b"] - ref_final["b"]).max(),
        np.abs(decoded["leaf_logits"] - ref_final["leaf_logits"]).max(),
    )

    train_pred = plaintext_predict(X_train, decoded)
    test_pred = plaintext_predict(X_test, decoded)
    train_acc = (train_pred == y_train).mean()
    test_acc = (test_pred == y_test).mean()

    result = {"train_acc": float(train_acc), "test_acc": float(test_acc), "max_err": float(max_err)}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
