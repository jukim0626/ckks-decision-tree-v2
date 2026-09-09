"""finalize_worker_N.py와 같은 지위 - 학습 끝난 session_dir의 최종 파라미터를 decrypt해서
train/test accuracy + plaintext 대비 오차를 계산. TreeConfig에 맞는
predict_variant/train_depthN_variant를 사용."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import load_scaled_dataset_subset  # noqa: E402
from core.data.serialization import load_context  # noqa: E402
from core.ckks_engine import create_bootstrap_engine  # noqa: E402
from models.gradient_soft_tree.depthN_ckks import decrypt_params_N  # noqa: E402
from models.gradient_soft_tree.opt.config import TreeConfig  # noqa: E402
from models.gradient_soft_tree.opt.reference_variants import predict_variant, train_depthN_variant  # noqa: E402


def main() -> None:
    session_dir = Path(sys.argv[1])
    config = json.loads((session_dir / "config.json").read_text())
    exp = json.loads((session_dir / "experiment_config.json").read_text())
    tree_config = TreeConfig(**exp)

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
        "alpha": [engine.read_ciphertext(params_dir / f"alpha_{i}.ct") for i in range(n_internal)],
        "threshold": [
            [engine.read_ciphertext(params_dir / f"threshold_{i}_{j}.ct") for j in range(n_features)]
            for i in range(n_internal)
        ],
        "leaf_logits": [engine.read_ciphertext(params_dir / f"leaf_{l}.ct") for l in range(n_leaves)],
    }
    decoded = decrypt_params_N(ctx, final_params, n_features, n_classes, depth)
    decoded["config"] = tree_config

    X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name)
    ref_final = train_depthN_variant(
        X_train, np.eye(n_classes)[y_train], depth=depth, config=tree_config, lr=lr, epochs=n_epochs, seed=seed
    )
    max_err = max(
        np.abs(decoded["alpha"] - ref_final["alpha"]).max(),
        np.abs(decoded["threshold"] - ref_final["threshold"]).max(),
        np.abs(decoded["leaf_logits"] - ref_final["leaf_logits"]).max(),
    )

    train_pred = predict_variant(X_train, decoded)
    test_pred = predict_variant(X_test, decoded)
    train_acc = (train_pred == y_train).mean()
    test_acc = (test_pred == y_test).mean()

    result = {"train_acc": float(train_acc), "test_acc": float(test_acc), "max_err": float(max_err)}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
