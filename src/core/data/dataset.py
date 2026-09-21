"""Encrypted training data 준비. sklearn은 dataset load/split/scaling에만 사용.
모든 모델 계보(baseline/opt/packed/local_loss, archive의 closed_form_mgi/client_assisted)가
공유하는 프리미티브 - 특정 프로토콜에 종속되지 않는다.

2026-09-09 리팩터로 client_assisted/dataset.py에서 분리됨(ctx 타입힌트가 client_assisted
전용 EncryptedTrainingContext였던 것만 제거 - 어떤 bootstrap context든 `.engine`/`.pk`만
있으면 동작한다)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def _load_pima_diabetes():
    """OpenML data_id=37(Pima Indians Diabetes, binary, 768 samples/8 feature)를
    다른 sklearn toy dataset과 같은 Bunch 인터페이스(.data/.target/.target_names)로 감싼다."""
    raw = fetch_openml(data_id=37, as_frame=False)
    target_names = sorted(set(raw.target))  # ['tested_negative', 'tested_positive']
    name_to_idx = {name: idx for idx, name in enumerate(target_names)}
    target = np.array([name_to_idx[label] for label in raw.target], dtype=int)
    return SimpleNamespace(data=raw.data, target=target, target_names=target_names)


def _load_soybean():
    """OpenML data_id=1023(soybean, binary N/P 축소판, 683 samples/35 categorical feature).
    35개 feature 전부 순서 없는 범주형 문자열(예: precip=gt-norm/norm/lt-norm)이다.

    2026-09-16: 처음엔 one-hot(35->99 feature)으로 풀었으나, depth=3(internal node 7개)
    기준 alpha/threshold 파라미터 ciphertext가 693개씩(총 1386개)로 늘어나
    breast_cancer depth=5(930개, setup 단계 즉사)보다도 많아져서 packed baseline으로도
    CUDA OOM(첫 노드의 attention softmax bootstrap에서 즉사) - packing은 파라미터
    *저장* 개수를 줄이는 게 아니라 샘플축 연산 *횟수*만 줄이는 거라 이 문제엔 무력함.
    사용자와 논의 후 **정수 라벨 인코딩**(카테고리를 pd.Categorical codes로 0,1,2...에
    매핑, feature 수는 35 그대로 유지)으로 전환 - 파라미터가 693*2/3=245개씩(35*7)으로
    줄어 OOM 위험이 낮아지는 대신, 카테고리 사이에 없던 순서/거리 관계를 sigmoid
    threshold가 학습 과정에서 임의로 부여하게 되는 트레이드오프를 감수한 것."""
    raw = fetch_openml(data_id=1023, as_frame=True)
    X = np.column_stack([raw.data[col].cat.codes.to_numpy(dtype=float) for col in raw.data.columns])
    target_names = sorted(set(raw.target))  # ['N', 'P']
    name_to_idx = {name: idx for idx, name in enumerate(target_names)}
    target = np.array([name_to_idx[label] for label in raw.target], dtype=int)
    return SimpleNamespace(data=X, target=target, target_names=target_names)


_DATASET_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
    "diabetes": _load_pima_diabetes,
    "soybean": _load_soybean,
}


@dataclass
class EncryptedDataset:
    """slot packing된 encrypted training data."""

    enc_features: list
    enc_labels: list
    n_samples: int
    n_features: int
    n_classes: int


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    """client-side label을 암호화 전 one-hot matrix로 변환."""
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y] = 1.0
    return y_one_hot


def encrypt_dataset(ctx: Any, X: np.ndarray, y_one_hot: np.ndarray) -> EncryptedDataset:
    """client가 feature와 one-hot label을 slot packing해서 암호화. ctx는 `.engine`/`.pk`를
    가진 어떤 bootstrap context든(core.bootstrap.BootstrapTrainingContext 등) 받는다."""
    enc_features = [
        ctx.engine.encrypt(X[:, feature_idx].astype(float).tolist(), ctx.pk)
        for feature_idx in range(X.shape[1])
    ]
    enc_labels = [
        ctx.engine.encrypt(y_one_hot[:, class_idx].astype(float).tolist(), ctx.pk)
        for class_idx in range(y_one_hot.shape[1])
    ]
    return EncryptedDataset(
        enc_features=enc_features,
        enc_labels=enc_labels,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        n_classes=y_one_hot.shape[1],
    )


def split_dataset_subset(
    dataset_name: str = "iris",
    test_size: float | int = 0.2,
    max_train: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """`load_scaled_dataset_subset`에서 scaling만 뺀 raw train/test split.

    2026-09-21: 학습 때 fit한 scaler를 세션 파일로 저장해서 inference에서 재사용하려면
    (재적합 없이) 이 함수로 raw split을 얻은 뒤 `fit_scaler`/`save_scaler`/`load_scaler`를
    따로 조합해서 쓴다 - `load_scaled_dataset_subset`처럼 매번 새로 fit하지 않는다.
    `random_state=42`로 고정돼 있어 같은 (dataset_name, test_size, max_train)이면 항상
    같은 split을 낸다."""
    if dataset_name not in _DATASET_LOADERS:
        raise ValueError(
            f"unknown dataset: {dataset_name!r}. choose from {sorted(_DATASET_LOADERS)}"
        )
    data = _DATASET_LOADERS[dataset_name]()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=42,
        stratify=data.target,
    )
    if max_train is not None and max_train < len(y_train):
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=max_train,
            random_state=42,
            stratify=y_train,
        )
    class_names = [str(name) for name in data.target_names]
    return X_train, X_test, y_train, y_test, class_names


def fit_scaler(X_train_raw: np.ndarray, feature_range: tuple[float, float] = (-1.0, 1.0)) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(X_train_raw)
    return scaler


def load_scaled_dataset_subset(
    dataset_name: str = "iris",
    test_size: float | int = 0.2,
    max_train: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """지정한 sklearn dataset을 train/test로 나눠서 쓴다 (train 기준 minmax_minus1_1
    scaling). test_size는 sklearn train_test_split 그대로: 0<x<1이면 비율(기본 0.2 ->
    80/20 split), 정수면 절대 개수(과거 기본값이었던 고정 30개 방식 - 데이터셋마다 train
    비율이 들쭉날쭉해지는 문제가 있어 비율 기본값으로 전환함).

    max_train을 주면 train을 그 개수로 stratified subsample한다.

    2026-09-21: 내부적으로 `split_dataset_subset`+`fit_scaler`로 나뉘었을 뿐 반환값/동작은
    이전과 동일하다 - scaler를 세션 간에 유지할 필요가 없는 호출부(plaintext reference
    비교, 1회성 디버그 스크립트 등)는 계속 이 함수를 그대로 쓰면 된다. **주의**: 이 함수는
    호출할 때마다 scaler를 새로 fit한다 - 학습 때와 다른 프로세스(finalize/predict)에서
    "같은 스케일링"이 보장돼야 하는 경우에는 이 함수 대신 학습 때 저장한 scaler를
    `load_scaler`로 복원해서 `.transform()`만 호출할 것(재적합 금지)."""
    X_train, X_test, y_train, y_test, class_names = split_dataset_subset(dataset_name, test_size, max_train)
    scaler = fit_scaler(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test, class_names


SCALER_SCHEMA_VERSION = 1


def save_scaler(scaler: MinMaxScaler, path: Path, *, dataset_name: str) -> None:
    """학습 때 fit한 scaler를 client 측 preprocessing 산출물로 저장한다 - 호출부는 이걸
    session_dir의 서버 쪽 자료(keys/dataset/params)와 분리된 위치(예: session_dir/client/)에
    두는 걸 권장(이 함수 자체가 경로를 강제하지는 않음). pickle/joblib 대신 JSON으로,
    `.transform()` 복원에 필요한 값(min_/scale_)과 검증용 메타데이터(dataset_name,
    feature 수, feature_range, data_min_/data_max_)만 남긴다."""
    payload = {
        "schema_version": SCALER_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "n_features": int(scaler.n_features_in_),
        "feature_range": list(scaler.feature_range),
        "data_min_": scaler.data_min_.tolist(),
        "data_max_": scaler.data_max_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "min_": scaler.min_.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def load_scaler(path: Path, *, expected_dataset_name: str, expected_n_features: int) -> MinMaxScaler:
    """`save_scaler`로 저장한 JSON을 복원한다(재적합 없음 - transform 전용).
    dataset_name/feature 수가 지금 이 프로세스가 기대하는 값과 다르면(설정 실수로 다른
    데이터셋의 scaler를 잘못 로드하는 사고를 막기 위해) 조용히 넘어가지 않고 즉시
    실패한다. 파일 자체가 없으면(scaler를 저장하지 않던 예전 세션) 명확한 안내와 함께
    실패한다 - 값을 추측해서 조용히 다른 preprocessing으로 넘어가지 않는다."""
    if not path.exists():
        raise FileNotFoundError(
            f"scaler 파일이 없습니다: {path}. 이 session은 scaler를 저장하지 않는 예전 코드로 "
            "만들어졌을 가능성이 높습니다 - migrate_scaler.py로 재구성하거나(원본 데이터/"
            "sklearn 버전이 학습 당시와 같다는 전제 하에만 유효하고, 정확한 복원을 보장하지 "
            "않음) 세션을 다시 학습하세요."
        )
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SCALER_SCHEMA_VERSION:
        raise ValueError(f"지원하지 않는 scaler schema_version: {payload.get('schema_version')!r}")
    if payload["dataset_name"] != expected_dataset_name:
        raise ValueError(
            f"scaler dataset_name 불일치: 파일={payload['dataset_name']!r} vs 기대값={expected_dataset_name!r}"
        )
    if payload["n_features"] != expected_n_features:
        raise ValueError(
            f"scaler n_features 불일치: 파일={payload['n_features']} vs 기대값={expected_n_features}"
        )
    scaler = MinMaxScaler(feature_range=tuple(payload["feature_range"]))
    scaler.n_features_in_ = payload["n_features"]
    scaler.data_min_ = np.array(payload["data_min_"])
    scaler.data_max_ = np.array(payload["data_max_"])
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.scale_ = np.array(payload["scale_"])
    scaler.min_ = np.array(payload["min_"])
    return scaler


def resolve_leaf_family_test_size(config: dict) -> float | int:
    """baseline/packed/opt(=leaf_logits 파라미터 스키마) 세션의 test_size를 안전하게
    복원한다. local_loss(=local_logits 스키마)에는 안 쓴다 - local_loss/setup_worker.py는
    test_size를 CLI로 받은 적이 없어 언제 만든 세션이든 항상 0.2였으므로(`config.get
    ("test_size", 0.2)`로 충분, 이건 추측이 아니라 실제 과거 동작) 이 함수가 필요 없다.

    baseline/packed/opt 계열은 2026-09-18부터 config.json에 test_size를 명시적으로
    저장한다. 그 이전 세션은 이 키 자체가 없는데, 그 시점 `load_scaled_dataset_subset`의
    실제 기본값이 test_size=30(절대 개수)이었다는 게 git 이력으로 확인되는 사실이라
    이 값을 쓴다 - 지금 코드의 기본값(0.2)으로 넘겨짚지 않는다."""
    if "test_size" in config:
        return config["test_size"]
    print(
        "[경고] config.json에 test_size가 없습니다 - 2026-09-18 이전 세션으로 보고 "
        "그 시점 코드의 실제 기본값(test_size=30, 절대 개수)을 적용합니다.",
        file=sys.stderr,
    )
    return 30
