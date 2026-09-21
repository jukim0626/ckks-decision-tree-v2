"""학습 끝난 session_dir의 최종 파라미터를 decrypt해서 train/test accuracy를 계산하는
전용 프로세스 (검증/평가 목적 - protocol 일부 아님). baseline/finalize_worker.py와
동일한 구조, local_logits의 레벨별 중첩 구조만 다르다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import load_scaler, split_dataset_subset  # noqa: E402
from core.data.serialization import load_context  # noqa: E402
from core.ckks_engine import create_bootstrap_engine  # noqa: E402
from models.gradient_soft_tree.local_loss.tree_ops import decrypt_params_N  # noqa: E402
from models.gradient_soft_tree.local_loss.reference import predict as plaintext_predict, train_depthN_local_loss  # noqa: E402
from models.gradient_soft_tree.params import load_alpha, load_local_logits, load_threshold  # noqa: E402


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

    engine = create_bootstrap_engine(
        mode=config["mode"], device_id=config["device_id"], level_preset=config.get("level_preset")
    )
    ctx = load_context(engine, session_dir / "keys", mode=config["mode"], device_id=config["device_id"])

    params_dir = session_dir / "params"
    final_params = {
        "alpha": load_alpha(engine, params_dir, n_internal),
        "threshold": load_threshold(engine, params_dir, n_internal, n_features),
        "local_logits": load_local_logits(engine, params_dir, depth),
    }
    decoded = decrypt_params_N(ctx, final_params, n_features, n_classes, depth)

    # 2026-09-21: local_loss는 test_size를 CLI로 받은 적이 없어 config에 없어도 0.2가
    # 곧 실제 과거 동작이다(baseline과 달리 resolve_leaf_family_test_size 불필요). 학습
    # 때 저장해둔 scaler로 transform만 한다(재적합 없음) - baseline/finalize_worker.py와
    # 동일한 이유.
    X_train_raw, X_test_raw, y_train, y_test, _ = split_dataset_subset(
        dataset_name, test_size=config.get("test_size", 0.2), max_train=config.get("max_train")
    )
    scaler = load_scaler(
        session_dir / "client" / "scaler.json", expected_dataset_name=dataset_name, expected_n_features=n_features
    )
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    ref_final = train_depthN_local_loss(
        X_train, np.eye(n_classes)[y_train], depth=depth, lr=lr, epochs=n_epochs, seed=seed
    )
    max_err = max(
        np.abs(decoded["alpha"] - ref_final["alpha"]).max(),
        np.abs(decoded["threshold"] - ref_final["threshold"]).max(),
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
