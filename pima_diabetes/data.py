"""Pima Indians Diabetes dataset loading and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
import ssl

import numpy as np


DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

FEATURE_NAMES = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
]

ZERO_AS_MISSING_FEATURES = [
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
]


@dataclass(frozen=True)
class PreprocessStats:
    """전처리에 필요한 통계값 묶음."""

    medians: np.ndarray
    means: np.ndarray
    stds: np.ndarray


def dataset_path() -> Path:
    """로컬 CSV 저장 경로를 반환합니다."""
    return Path(__file__).resolve().parent / "pima-indians-diabetes.csv"


def download_dataset(path: Path | None = None) -> Path:
    """Pima Indians Diabetes CSV를 로컬 파일로 다운로드합니다."""
    target_path = path or dataset_path()
    if target_path.exists():
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(DATA_URL) as response:
            target_path.write_bytes(response.read())
    except URLError:
        # macOS Python에서 local issuer certificate 문제가 날 때만 fallback
        unverified_context = ssl._create_unverified_context()
        with urlopen(DATA_URL, context=unverified_context) as response:
            target_path.write_bytes(response.read())
    return target_path


def load_raw_dataset(path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """CSV에서 raw feature matrix X와 label y를 읽습니다."""
    csv_path = path or dataset_path()
    if not csv_path.exists():
        csv_path = download_dataset(csv_path)

    data = np.loadtxt(csv_path, delimiter=",")
    x_raw = data[:, :-1].astype(float)
    y = data[:, -1].astype(int)
    return x_raw, y


def fit_preprocess_stats(x_train_raw: np.ndarray) -> PreprocessStats:
    """train split 기준 median imputation과 standardization 통계를 계산합니다."""
    x_with_nan = x_train_raw.astype(float).copy()

    for feature_name in ZERO_AS_MISSING_FEATURES:
        feature_idx = FEATURE_NAMES.index(feature_name)
        x_with_nan[x_with_nan[:, feature_idx] == 0.0, feature_idx] = np.nan

    medians = np.nanmedian(x_with_nan, axis=0)
    x_imputed = np.where(np.isnan(x_with_nan), medians, x_with_nan)

    means = np.mean(x_imputed, axis=0)
    stds = np.std(x_imputed, axis=0)
    stds = np.where(stds == 0.0, 1.0, stds)

    return PreprocessStats(medians=medians, means=means, stds=stds)


def transform_features(x_raw: np.ndarray, stats: PreprocessStats) -> np.ndarray:
    """raw feature를 median imputation 후 z-score로 변환합니다."""
    x_with_nan = x_raw.astype(float).copy()

    for feature_name in ZERO_AS_MISSING_FEATURES:
        feature_idx = FEATURE_NAMES.index(feature_name)
        x_with_nan[x_with_nan[:, feature_idx] == 0.0, feature_idx] = np.nan

    x_imputed = np.where(np.isnan(x_with_nan), stats.medians, x_with_nan)
    return (x_imputed - stats.means) / stats.stds


def standardized_threshold_to_raw(
    feature_idx: int,
    threshold: float,
    stats: PreprocessStats,
) -> float:
    """standardized threshold를 원래 feature 단위 threshold로 되돌립니다."""
    return float(threshold * stats.stds[feature_idx] + stats.means[feature_idx])
