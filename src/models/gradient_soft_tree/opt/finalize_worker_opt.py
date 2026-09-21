"""finalize_worker_N.py와 같은 지위 - 학습 끝난 session_dir의 최종 파라미터를 decrypt해서
train/test accuracy + plaintext 대비 오차를 계산. TreeConfig에 맞는
predict_variant/train_depthN_variant를 사용."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import load_scaler, resolve_leaf_family_test_size, split_dataset_subset  # noqa: E402
from core.data.serialization import load_context  # noqa: E402
from core.ckks_engine import create_bootstrap_engine  # noqa: E402
from models.gradient_soft_tree.baseline.tree_ops import decrypt_params_N  # noqa: E402
from models.gradient_soft_tree.opt.config import TreeConfig  # noqa: E402
from models.gradient_soft_tree.opt.reference_variants import predict_variant, train_depthN_variant  # noqa: E402
from models.gradient_soft_tree.params import load_alpha, load_leaf_logits, load_threshold  # noqa: E402


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
        "alpha": load_alpha(engine, params_dir, n_internal),
        "threshold": load_threshold(engine, params_dir, n_internal, n_features),
        "leaf_logits": load_leaf_logits(engine, params_dir, n_leaves),
    }
    decoded = decrypt_params_N(ctx, final_params, n_features, n_classes, depth)
    decoded["config"] = tree_config

    # 2026-09-21: opt는 baseline/setup_worker.py를 그대로 재사용하므로 config 스키마도
    # baseline과 같다(leaf_logits 계열) - test_size/scaler를 안 넘겨서 학습 때와 다른
    # preprocessing으로 평가할 위험을 baseline/finalize_worker.py와 동일하게 없앤다.
    # (현재 opt/train_opt.py는 setup_worker를 항상 기본값으로만 부르므로 지금 당장
    # 재현되는 활성 버그는 아니지만, 구조는 baseline/packed와 동일한 잠재 버그였다.)
    test_size = resolve_leaf_family_test_size(config)
    X_train_raw, X_test_raw, y_train, y_test, _ = split_dataset_subset(
        dataset_name, test_size=test_size, max_train=config.get("max_train")
    )
    scaler = load_scaler(
        session_dir / "client" / "scaler.json", expected_dataset_name=dataset_name, expected_n_features=n_features
    )
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
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
