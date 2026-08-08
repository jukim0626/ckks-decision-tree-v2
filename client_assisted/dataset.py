"""Encrypted training data 준비. sklearn은 dataset load/split/scaling에만 사용."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from client_assisted.context import EncryptedTrainingContext


def _load_pima_diabetes():
    """OpenML data_id=37(Pima Indians Diabetes, binary, 768 samples/8 feature)를
    다른 sklearn toy dataset과 같은 Bunch 인터페이스(.data/.target/.target_names)로 감싼다."""
    raw = fetch_openml(data_id=37, as_frame=False)
    target_names = sorted(set(raw.target))  # ['tested_negative', 'tested_positive']
    name_to_idx = {name: idx for idx, name in enumerate(target_names)}
    target = np.array([name_to_idx[label] for label in raw.target], dtype=int)
    return SimpleNamespace(data=raw.data, target=target, target_names=target_names)


_DATASET_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
    "diabetes": _load_pima_diabetes,
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


def encrypt_dataset(
    ctx: EncryptedTrainingContext,
    X: np.ndarray,
    y_one_hot: np.ndarray,
) -> EncryptedDataset:
    """client가 feature와 one-hot label을 slot packing해서 암호화."""
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
    test_size: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """지정한 sklearn dataset에서 test_size(절대 개수)만 test로 떼어내고 나머지 전체를
    train으로 쓴다 (train 기준 minmax_minus1_1 scaling).

    train은 dataset 전체(- test_size)를 쓴다 (iris=150 -> train 120, wine=178 -> train
    148, breast_cancer=569 -> train 539). test는 기본 30개 고정.

    encrypted training은 CKKS SIMD packing 덕분에 train sample 수가 늘어나도 학습
    시간이 거의 안 늘어나지만 (candidate당 sigmoid 계산은 packing된 벡터 전체에 1번),
    encrypted inference는 test sample마다 순차적으로 돌기 때문에 test 수에 비례해서
    시간이 늘어난다 - 그래서 test는 고정된 작은 개수로 두고 train만 전체를 쓴다.
    feature 수가 많은 dataset(예: breast_cancer, 30 feature)은 candidate 총량도 커지므로
    candidate_count나 --max-level을 낮춰서 시도하는 걸 권장한다.
    """
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
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    class_names = [str(name) for name in data.target_names]
    return X_train_scaled, X_test_scaled, y_train, y_test, class_names
