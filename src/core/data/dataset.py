"""Encrypted training data 준비. sklearn은 dataset load/split/scaling에만 사용.
모든 모델 계보(baseline/opt/packed/local_loss, archive의 closed_form_mgi/client_assisted)가
공유하는 프리미티브 - 특정 프로토콜에 종속되지 않는다.

2026-09-09 리팩터로 client_assisted/dataset.py에서 분리됨(ctx 타입힌트가 client_assisted
전용 EncryptedTrainingContext였던 것만 제거 - 어떤 bootstrap context든 `.engine`/`.pk`만
있으면 동작한다)."""

from __future__ import annotations

from dataclasses import dataclass
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


def load_scaled_dataset_subset(
    dataset_name: str = "iris",
    test_size: float | int = 0.2,
    max_train: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """지정한 sklearn dataset을 train/test로 나눠서 쓴다 (train 기준 minmax_minus1_1
    scaling). test_size는 sklearn train_test_split 그대로: 0<x<1이면 비율(기본 0.2 ->
    80/20 split), 정수면 절대 개수(과거 기본값이었던 고정 30개 방식 - 데이터셋마다 train
    비율이 들쭉날쭉해지는 문제가 있어 비율 기본값으로 전환함).

    max_train을 주면 train을 그 개수로 stratified subsample한다."""
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
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    class_names = [str(name) for name in data.target_names]
    return X_train_scaled, X_test_scaled, y_train, y_test, class_names
