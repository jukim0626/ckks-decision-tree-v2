"""학습 끝난 session_dir의 최종 파라미터를 decrypt해서 train/test accuracy를 계산하는
전용 프로세스 (검증/평가 목적 - protocol 일부 아님, closed_form_mgi의 debug_decrypt_winner와
같은 지위). 오케스트레이터(train_depthN_ckks.py) 자신이 GPU를 잡지 않도록 이것도 별도
프로세스로 뺐다 (setup_worker_N.py와 같은 이유)."""

from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import load_scaler, resolve_leaf_family_test_size, scale_features, split_dataset_subset  # noqa: E402
from core.data.serialization import load_client_secret_key, load_server_context  # noqa: E402
from core.ckks_engine import create_bootstrap_engine  # noqa: E402
from models.gradient_soft_tree.baseline.tree_ops import decrypt_params_N  # noqa: E402
from models.gradient_soft_tree.baseline.reference import domain_report, predict as plaintext_predict, train_depthN  # noqa: E402
from models.gradient_soft_tree.params import load_alpha, load_leaf_logits, load_threshold  # noqa: E402


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
    ctx = load_server_context(engine, session_dir / "keys", mode=config["mode"], device_id=config["device_id"])
    ctx = replace(ctx, sk=load_client_secret_key(engine, session_dir))  # client 측 검증용 decrypt

    params_dir = session_dir / "params"
    final_params = {
        "alpha": load_alpha(engine, params_dir, n_internal),
        "threshold": load_threshold(engine, params_dir, n_internal, n_features),
        "leaf_logits": load_leaf_logits(engine, params_dir, n_leaves),
    }
    decoded = decrypt_params_N(ctx, final_params, n_features, n_classes, depth)

    # 2026-09-21: 학습 때와 같은 test_size/max_train으로 raw split을 재현하고(구세션도
    # resolve_leaf_family_test_size가 그 시점 실제 기본값을 적용), 학습 때 저장해둔
    # scaler로 transform만 한다(재적합 없음) - finalize가 자기만의 scaler를 새로 fit해서
    # 학습 때와 다른 스케일링으로 평가하는 사고를 막는다.
    test_size = resolve_leaf_family_test_size(config)
    X_train_raw, X_test_raw, y_train, y_test, _ = split_dataset_subset(
        dataset_name, test_size=test_size, max_train=config.get("max_train")
    )
    scaler = load_scaler(
        session_dir / "client" / "scaler.json", expected_dataset_name=dataset_name, expected_n_features=n_features
    )
    X_train = scale_features(scaler, X_train_raw)
    X_test = scale_features(scaler, X_test_raw)
    # 2026-10-07: 오차/정확도를 두 기준으로 분리한다.
    # - true: 진짜 sigmoid/exp plaintext 모델 기준 (= 다항식 근사 오차 + CKKS 노이즈)
    # - poly: CKKS와 같은 다항식을 쓴 plaintext 모델 기준 (= 순수 CKKS 노이즈)
    # 세 번째 기준(진짜 encrypted inference)은 packed/predict_packed.py가 따로 잰다.
    # 여기 정확도는 전부 "decrypt한 파라미터를 평문 predict에 넣은 값"이라 protocol 일부가 아니다.
    def _max_param_err(ref: dict) -> float:
        return float(max(
            np.abs(decoded["alpha"] - ref["alpha"]).max(),
            np.abs(decoded["threshold"] - ref["threshold"]).max(),
            np.abs(decoded["leaf_logits"] - ref["leaf_logits"]).max(),
        ))

    y_train_oh = np.eye(n_classes)[y_train]
    ref_true = train_depthN(X_train, y_train_oh, depth=depth, lr=lr, epochs=n_epochs, seed=seed, approx="true")
    ref_poly = train_depthN(X_train, y_train_oh, depth=depth, lr=lr, epochs=n_epochs, seed=seed, approx="poly")

    result = {
        # 하위호환 키(기존 로그/스크립트가 읽음): true 기준
        "train_acc": float((plaintext_predict(X_train, decoded, approx="true") == y_train).mean()),
        "test_acc": float((plaintext_predict(X_test, decoded, approx="true") == y_test).mean()),
        "max_err": _max_param_err(ref_true),
        # poly 기준: CKKS가 실제로 계산하는 함수
        "train_acc_poly": float((plaintext_predict(X_train, decoded, approx="poly") == y_train).mean()),
        "test_acc_poly": float((plaintext_predict(X_test, decoded, approx="poly") == y_test).mean()),
        "max_err_poly": _max_param_err(ref_poly),
        # 다항식 근사 구간 이탈 진단
        "domain_train": domain_report(X_train, decoded),
        "domain_test": domain_report(X_test, decoded),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
