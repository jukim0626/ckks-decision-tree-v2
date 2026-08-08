"""공개(public) split threshold grid 생성. 데이터를 보지 않고 미리 정해두는 후보들."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PublicSplitCandidate:
    """공개 split 후보."""

    feature_idx: int
    threshold: float


def make_public_grid_candidates(
    n_features: int,
    thresholds: np.ndarray,
) -> list[PublicSplitCandidate]:
    """public threshold grid로 split 후보를 만든다."""
    candidates = []
    for feature_idx in range(n_features):
        for threshold in thresholds:
            candidates.append(
                PublicSplitCandidate(
                    feature_idx=feature_idx,
                    threshold=float(threshold),
                )
            )
    return candidates


def make_small_public_threshold_grid(
    candidate_count: int = 3,
    feature_range: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """작은 public threshold grid를 생성."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be at least 1")
    low, high = feature_range
    if candidate_count == 1:
        return np.array([(low + high) / 2.0], dtype=float)
    return np.linspace(low, high, candidate_count + 2, dtype=float)[1:-1]
