"""검증 전용 plaintext 함수들. protocol core가 아니라 client가 이미 아는 평문 threshold와
debug decrypt한 leaf_counts로 정확도를 확인하는 용도."""

from __future__ import annotations

import numpy as np

from client_assisted.client_ops import ClientSelectedSplit, client_decrypt_scalar
from client_assisted.context import EncryptedTrainingContext
from client_assisted.models import (
    ClientFixedDepthSelections,
    EncryptedFixedDepthTreeModel,
)


def plaintext_split_weights(X: np.ndarray, selected: ClientSelectedSplit) -> np.ndarray:
    """검증용 plaintext selected split right weight 계산."""
    from sigmoid_approx_coeffs import SPLIT_SIGMOID_COEFFS

    diff = X[:, selected.feature_idx] - selected.threshold
    return np.polynomial.polynomial.polyval(diff, SPLIT_SIGMOID_COEFFS)


def plaintext_fixed_depth_leaf_weights(
    X: np.ndarray,
    selections: ClientFixedDepthSelections,
) -> list[np.ndarray]:
    """검증용 plaintext fixed-depth soft traversal leaf weights 계산."""
    leaf_weights: list[np.ndarray] = []
    selection_idx = 0

    def walk(node_weights: np.ndarray, current_depth: int) -> None:
        nonlocal selection_idx
        if current_depth == selections.depth:
            leaf_weights.append(node_weights)
            return

        selected = selections.selections[selection_idx]
        selection_idx += 1
        right_weights = plaintext_split_weights(X, selected)
        left_weights = 1.0 - right_weights
        walk(node_weights * left_weights, current_depth + 1)
        walk(node_weights * right_weights, current_depth + 1)

    walk(np.ones(X.shape[0], dtype=float), current_depth=0)
    return leaf_weights


def predict_fixed_depth_soft_plaintext(
    X: np.ndarray,
    selections: ClientFixedDepthSelections,
    leaf_counts: list[np.ndarray],
) -> np.ndarray:
    """검증용 plaintext soft traversal로 fixed-depth tree 예측 (leaf-count score 방식).

    predict_depth2_soft_plaintext의 일반화 버전. encrypted_fixed_depth_inference.py의
    encrypted_traverse_and_predict_fixed_depth()와 같은 leaf class count weighted-sum
    방식이라 이 함수와 비교해야 한다.
    """
    leaf_weights = plaintext_fixed_depth_leaf_weights(X, selections)
    scores = np.zeros((X.shape[0], leaf_counts[0].shape[0]), dtype=float)
    for weights, counts in zip(leaf_weights, leaf_counts):
        scores += weights[:, None] * counts[None, :]
    return np.argmax(scores, axis=1)


def predict_fixed_depth_leaf_majority_plaintext(
    X: np.ndarray,
    selections: ClientFixedDepthSelections,
    leaf_counts: list[np.ndarray],
) -> np.ndarray:
    """검증용으로 fixed-depth tree의 leaf majority class 예측."""
    leaf_weights = plaintext_fixed_depth_leaf_weights(X, selections)
    scores = np.zeros((X.shape[0], leaf_counts[0].shape[0]), dtype=float)
    for weights, counts in zip(leaf_weights, leaf_counts):
        leaf_class = int(np.argmax(counts))
        scores[:, leaf_class] += weights
    return np.argmax(scores, axis=1)


def debug_decrypt_fixed_depth_leaf_counts(
    ctx: EncryptedTrainingContext,
    model: EncryptedFixedDepthTreeModel,
) -> list[np.ndarray]:
    """검증용으로 fixed-depth encrypted leaf counts를 client가 decrypt."""
    decrypted_counts = []
    for leaf_count in model.leaf_counts:
        decrypted_counts.append(
            np.array(
                [client_decrypt_scalar(ctx, count) for count in leaf_count],
                dtype=float,
            )
        )
    return decrypted_counts


def format_float_list(values: np.ndarray) -> list[float]:
    """출력용으로 numpy scalar를 일반 float list로 변환."""
    return [round(float(value), 6) for value in values]
