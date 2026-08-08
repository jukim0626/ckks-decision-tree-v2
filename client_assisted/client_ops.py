"""Client(KMS)가 수행하는 연산: encrypted aggregate decrypt, weighted Gini argmin, 재encrypt.

논문 protocol에서 client만 할 수 있는 부분 (secret key를 갖고 있어야 하는 연산)만 모아둔다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from client_assisted.context import EncryptedTrainingContext


@dataclass
class ClientSelectedSplit:
    """client가 Gini argmin으로 선택한 split."""

    candidate_index: int
    feature_idx: int
    threshold: float
    gini_score: float


@dataclass
class EncryptedSelectedSplit:
    """client가 다시 encrypt해서 server에 보내는 selected split."""

    enc_feature_masks: list
    enc_threshold: object


def client_decrypt_scalar(ctx: EncryptedTrainingContext, ciphertext) -> float:
    """client-side scalar decrypt."""
    return float(ctx.engine.decrypt(ciphertext, ctx.sk)[0])


def client_decrypt_aggregate_counts(
    ctx: EncryptedTrainingContext,
    aggregate,
) -> tuple[np.ndarray, np.ndarray]:
    """client가 encrypted aggregate를 decrypt해 class count vector로 변환."""
    left_counts = np.array(
        [client_decrypt_scalar(ctx, count) for count in aggregate.left_counts],
        dtype=float,
    )
    right_counts = np.array(
        [client_decrypt_scalar(ctx, count) for count in aggregate.right_counts],
        dtype=float,
    )
    return left_counts, right_counts


def weighted_gini_from_counts(class_counts: np.ndarray) -> float:
    """class count vector에서 weighted Gini 항을 계산."""
    total = float(np.sum(class_counts))
    if total <= 1e-12:
        return 0.0
    class_probs = class_counts / total
    gini = 1.0 - float(np.sum(class_probs * class_probs))
    return total * gini


def client_score_aggregate(ctx: EncryptedTrainingContext, aggregate) -> float:
    """client가 aggregate 하나를 decrypt해서 weighted Gini 점수만 계산.

    candidate를 스트리밍으로 처리할 때(server_compute_one_weighted_candidate_aggregate와
    짝) 호출부가 점수 확인 직후 aggregate를 바로 버릴 수 있도록 점수만 반환한다.
    """
    left_counts, right_counts = client_decrypt_aggregate_counts(ctx, aggregate)
    return weighted_gini_from_counts(left_counts) + weighted_gini_from_counts(
        right_counts
    )


def client_encrypt_selected_split(
    ctx: EncryptedTrainingContext,
    selected: ClientSelectedSplit,
    n_samples: int,
    n_features: int,
) -> EncryptedSelectedSplit:
    """client가 selected feature one-hot과 threshold를 vector로 다시 encrypt."""
    enc_feature_masks = []
    for feature_idx in range(n_features):
        value = 1.0 if feature_idx == selected.feature_idx else 0.0
        enc_feature_masks.append(ctx.engine.encrypt([value] * n_samples, ctx.pk))
    enc_threshold = ctx.engine.encrypt([selected.threshold] * n_samples, ctx.pk)
    return EncryptedSelectedSplit(
        enc_feature_masks=enc_feature_masks,
        enc_threshold=enc_threshold,
    )
