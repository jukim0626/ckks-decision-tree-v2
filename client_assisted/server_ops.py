"""Server가 수행하는 homomorphic 연산: candidate별 aggregate 계산, selected split 적용.

server는 secret key가 없으므로 이 파일의 함수들은 전부 ciphertext만 다루고 decrypt하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

from ckks_tree import sigmoid_approx_enc

from client_assisted.candidates import PublicSplitCandidate
from client_assisted.client_ops import EncryptedSelectedSplit
from client_assisted.context import EncryptedTrainingContext
from client_assisted.dataset import EncryptedDataset


@dataclass
class EncryptedAggregate:
    """candidate 하나에 대한 encrypted left/right class aggregate."""

    candidate: PublicSplitCandidate
    left_counts: list
    right_counts: list


def encrypt_constant_vector(
    ctx: EncryptedTrainingContext,
    value: float,
    n_samples: int,
) -> object:
    """같은 값을 n_samples개 slot에 채운 vector를 encrypt."""
    return ctx.engine.encrypt([value] * n_samples, ctx.pk)


def encrypted_slot_sum(ctx: EncryptedTrainingContext, enc_values) -> object:
    """ciphertext slot 전체 합을 ciphertext로 계산."""
    return ctx.engine.sum(enc_values, ctx.rotation_key)


def encrypted_class_counts(
    ctx: EncryptedTrainingContext,
    enc_weights,
    enc_labels: list,
) -> list:
    """encrypted weights와 encrypted one-hot label로 class별 count 계산."""
    counts = []
    for enc_label in enc_labels:
        weighted_label = ctx.engine.multiply(enc_weights, enc_label, ctx.rlk)
        counts.append(encrypted_slot_sum(ctx, weighted_label))
    return counts


def encrypted_weighted_class_counts(
    ctx: EncryptedTrainingContext,
    enc_node_weights,
    enc_branch_weights,
    enc_labels: list,
) -> list:
    """node weight와 branch weight를 곱한 뒤 class별 count 계산."""
    weighted_branch = ctx.engine.multiply(
        enc_node_weights,
        enc_branch_weights,
        ctx.rlk,
    )
    return encrypted_class_counts(ctx, weighted_branch, enc_labels)


def server_compute_one_weighted_candidate_aggregate(
    ctx: EncryptedTrainingContext,
    dataset: EncryptedDataset,
    candidate: PublicSplitCandidate,
    enc_node_weights,
) -> EncryptedAggregate:
    """server가 candidate 하나에 대한 encrypted aggregate만 계산.

    candidate 전체를 한 번에 다 계산해서 리스트로 쌓아두면(behavior 예전 버전) client가
    고르기 전까지 candidate 수만큼의 ciphertext가 전부 GPU에 동시에 살아있어야 해서
    candidate가 많을 때(예: 210개) OOM의 직접적인 원인이 된다. 이 함수는 candidate 하나만
    계산해서 즉시 client에게 넘기고, training.py의 호출부가 점수 확인 후 바로 버릴 수
    있게 한다 (peak 메모리를 O(candidates)에서 O(1)로 낮춤).
    """
    enc_feature = dataset.enc_features[candidate.feature_idx]
    enc_diff = ctx.engine.subtract(enc_feature, candidate.threshold)
    right_prob = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff)
    left_prob = ctx.engine.subtract(1.0, right_prob)
    left_counts = encrypted_weighted_class_counts(
        ctx,
        enc_node_weights,
        left_prob,
        dataset.enc_labels,
    )
    right_counts = encrypted_weighted_class_counts(
        ctx,
        enc_node_weights,
        right_prob,
        dataset.enc_labels,
    )
    return EncryptedAggregate(
        candidate=candidate,
        left_counts=left_counts,
        right_counts=right_counts,
    )


def server_selected_feature_value(
    ctx: EncryptedTrainingContext,
    dataset: EncryptedDataset,
    selected_split: EncryptedSelectedSplit,
) -> object:
    """server가 encrypted feature one-hot으로 selected feature vector 계산."""
    selected_feature = None
    for enc_feature, enc_mask in zip(
        dataset.enc_features,
        selected_split.enc_feature_masks,
    ):
        selected_piece = ctx.engine.multiply(enc_feature, enc_mask, ctx.rlk)
        if selected_feature is None:
            selected_feature = selected_piece
        else:
            selected_feature = ctx.engine.add(selected_feature, selected_piece)
    return selected_feature


def server_compute_child_weights(
    ctx: EncryptedTrainingContext,
    dataset: EncryptedDataset,
    selected_split: EncryptedSelectedSplit,
) -> tuple[object, object]:
    """server가 encrypted selected split으로 left/right child weights 계산."""
    selected_feature = server_selected_feature_value(ctx, dataset, selected_split)
    enc_diff = ctx.engine.subtract(selected_feature, selected_split.enc_threshold)
    right_weights = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff)
    left_weights = ctx.engine.subtract(1.0, right_weights)
    return left_weights, right_weights


def server_compute_weighted_child_weights(
    ctx: EncryptedTrainingContext,
    dataset: EncryptedDataset,
    selected_split: EncryptedSelectedSplit,
    enc_node_weights,
) -> tuple[object, object]:
    """node weight 아래에서 encrypted selected split의 left/right weights 계산."""
    left_branch, right_branch = server_compute_child_weights(
        ctx,
        dataset,
        selected_split,
    )
    left_weights = ctx.engine.multiply(enc_node_weights, left_branch, ctx.rlk)
    right_weights = ctx.engine.multiply(enc_node_weights, right_branch, ctx.rlk)
    return left_weights, right_weights
