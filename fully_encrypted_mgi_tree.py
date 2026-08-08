"""depth=1 stump(fully_encrypted_mgi_simd_argmin.py)를 임의 depth로 재귀 확장.

client decrypt 없이: 노드마다 encrypted node_weights로 가중된 candidate MGI score를
계산 -> SIMD argmin(sign_bootstrap)으로 최선의 split을 encrypted 상태로 찾음 ->
client_assisted.server_ops.server_compute_weighted_child_weights(이미 검증된 함수,
"선택된 split으로 자식 weight 계산"은 depth와 무관한 범용 CKKS 연산이라 그대로 재사용)로
자식 weight를 encrypted로 계산 -> 재귀.
"""

from __future__ import annotations

import gc
import time

from ckks_tree import sigmoid_approx_enc
from client_assisted.client_ops import EncryptedSelectedSplit
from client_assisted.dataset import EncryptedDataset
from client_assisted.server_ops import encrypted_class_counts, server_compute_weighted_child_weights
from fully_encrypted_mgi_stump import encrypted_mgi_from_counts, encrypted_weighted_class_counts, ensure_level
from fully_encrypted_mgi_simd_argmin import (
    evaluate_all_candidates_packed,
    next_power_of_two,
    simd_reduce_argmin,
)


def evaluate_candidate_weighted(ctx, dataset: EncryptedDataset, candidate, enc_node_weights):
    """evaluate_candidate()에 enc_node_weights(이 노드까지 도달한 soft membership)를 곱해서
    가중 MGI score/counts를 계산."""
    enc_feature = dataset.enc_features[candidate.feature_idx]
    enc_diff = ctx.engine.subtract(enc_feature, candidate.threshold)
    right_prob = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff)
    left_prob = ctx.engine.subtract(1.0, right_prob)
    right_prob = ctx.engine.multiply(right_prob, enc_node_weights, ctx.rlk)
    left_prob = ctx.engine.multiply(left_prob, enc_node_weights, ctx.rlk)
    left_counts = encrypted_weighted_class_counts(ctx, left_prob, dataset.enc_labels)
    right_counts = encrypted_weighted_class_counts(ctx, right_prob, dataset.enc_labels)
    score = ctx.engine.add(
        encrypted_mgi_from_counts(ctx, left_counts),
        encrypted_mgi_from_counts(ctx, right_counts),
    )
    return score, left_counts, right_counts


def evaluate_all_candidates_packed_weighted(
    ctx, dataset: EncryptedDataset, candidates: list, enc_node_weights, score_normalizer: float, verbose: bool
):
    """evaluate_all_candidates_packed()의 가중치 버전 (스트리밍 packing, O(1) peak 메모리)."""
    n = len(candidates)
    n_pow2 = next_power_of_two(n)
    n_features = dataset.n_features
    n_classes = dataset.n_classes

    from fully_encrypted_mgi_simd_argmin import scatter_to_slot
    import numpy as np

    packed_score = None
    packed_left = [None] * n_classes
    packed_right = [None] * n_classes
    for idx, candidate in enumerate(candidates):
        score, lc, rc = evaluate_candidate_weighted(ctx, dataset, candidate, enc_node_weights)
        piece_score = scatter_to_slot(ctx, score, idx)
        packed_score = piece_score if packed_score is None else ctx.engine.add(packed_score, piece_score)
        for c in range(n_classes):
            piece_l = scatter_to_slot(ctx, lc[c], idx)
            packed_left[c] = piece_l if packed_left[c] is None else ctx.engine.add(packed_left[c], piece_l)
            piece_r = scatter_to_slot(ctx, rc[c], idx)
            packed_right[c] = piece_r if packed_right[c] is None else ctx.engine.add(packed_right[c], piece_r)
        del score, lc, rc
        gc.collect()
        if verbose:
            print(f"    [scoring] candidate {idx + 1}/{n}", flush=True)

    thresholds_vec = [c.threshold for c in candidates] + [0.0] * (n_pow2 - n)
    packed_threshold = ctx.engine.encrypt(thresholds_vec, ctx.pk)
    packed_feature_indicators = []
    for f in range(n_features):
        vec = [1.0 if c.feature_idx == f else 0.0 for c in candidates] + [0.0] * (n_pow2 - n)
        packed_feature_indicators.append(ctx.engine.encrypt(vec, ctx.pk))

    if n_pow2 > n:
        pad_mask_vec = [0.0] * n + [1.0] * (n_pow2 - n)
        pad_mask = ctx.engine.encrypt(pad_mask_vec, ctx.pk)
        sentinel = ctx.engine.encrypt([score_normalizer] * n_pow2, ctx.pk)
        packed_score = ctx.engine.add(packed_score, ctx.engine.multiply(pad_mask, sentinel, ctx.rlk))

    aux = [packed_threshold] + packed_feature_indicators + packed_left + packed_right
    return packed_score, aux, n_pow2, n_features, n_classes


def find_best_split_weighted(
    ctx, dataset, candidates: list, enc_node_weights, score_normalizer: float, sharpen_iterations: int, verbose: bool
):
    """한 노드에서 enc_node_weights 기준 최선의 split을 encrypted로 찾아 EncryptedSelectedSplit 반환.

    호출부(train_node)가 이미 enc_node_weights를 ensure_level로 복구해서 넘겨준다고 가정
    (여기서 또 복구하면 train_node가 들고 있는 참조와 어긋난다 - 아래 train_node 주석 참고)."""
    packed_score, aux, n_pow2, n_features, n_classes = evaluate_all_candidates_packed_weighted(
        ctx, dataset, candidates, enc_node_weights, score_normalizer, verbose
    )
    _, winner_aux = simd_reduce_argmin(ctx, packed_score, aux, n_pow2, score_normalizer, sharpen_iterations, verbose)
    winner_threshold = winner_aux[0]
    winner_feature_selectors = winner_aux[1 : 1 + n_features]
    return EncryptedSelectedSplit(enc_feature_masks=winner_feature_selectors, enc_threshold=winner_threshold)


def train_fully_encrypted_mgi_tree(
    ctx,
    dataset: EncryptedDataset,
    candidates: list,
    depth: int,
    score_normalizer: float | None = None,
    sharpen_iterations: int | list[int] = 12,
    verbose: bool = True,
):
    """client decrypt 없이 depth(>=1)짜리 fixed-depth tree를 학습.

    반환: (node_splits: EncryptedSelectedSplit의 pre-order 리스트, leaf_counts: leaf별
    encrypted class count 리스트) - client_assisted.EncryptedFixedDepthTreeModel과
    같은 구조라 client_assisted의 encrypted inference(encrypted_traverse_and_predict_fixed_depth)를
    그대로 재사용할 수 있다.
    """
    if score_normalizer is None:
        score_normalizer = float(dataset.n_samples**2)
    if depth < 1:
        raise ValueError("depth must be at least 1")

    total_nodes = (1 << depth) - 1
    node_splits: list = []
    leaf_weights: list = []
    root_weights = ctx.engine.encrypt([1.0] * dataset.n_samples, ctx.pk)

    def train_node(enc_node_weights, current_depth: int) -> None:
        if current_depth == depth:
            leaf_weights.append(enc_node_weights)
            return
        # 부모에서 넘어온 enc_node_weights는 이미 selected-feature 계산+sigmoid+multiply로
        # level을 꽤 소모한 상태다 (server_compute_weighted_child_weights). 이 노드에서
        # sigmoid+multiply를 또 거치면(evaluate_candidate_weighted) level이 0 이하로
        # 떨어져서 곱셈 자체가 실패한다 ("the first input ciphertext should have a
        # positive level" - depth=3 root->child 전환에서 실제로 재현됨). 이 노드가 쓰는
        # 단일 참조를 여기서 한 번만 복구해서 find_best_split_weighted와
        # server_compute_weighted_child_weights 둘 다 같은(복구된) ciphertext를 쓰게 한다.
        enc_node_weights = ensure_level(ctx, enc_node_weights, min_level=16)
        node_idx = len(node_splits)  # pre-order index (0-based) - 노드별 sharpen 리스트와 맞춤
        this_sharpen = sharpen_iterations[node_idx] if isinstance(sharpen_iterations, list) else sharpen_iterations
        t0 = time.time()
        selected_split = find_best_split_weighted(
            ctx, dataset, candidates, enc_node_weights, score_normalizer, this_sharpen, verbose=False
        )
        node_splits.append(selected_split)
        elapsed = time.time() - t0
        print(
            f"[tree] node {len(node_splits)}/{total_nodes} (depth {current_depth}) done | "
            f"sharpen={this_sharpen} | {elapsed:.1f}s",
            flush=True,
        )
        left_weights, right_weights = server_compute_weighted_child_weights(
            ctx, dataset, selected_split, enc_node_weights
        )
        del enc_node_weights
        gc.collect()
        train_node(left_weights, current_depth + 1)
        del left_weights
        train_node(right_weights, current_depth + 1)

    train_node(root_weights, current_depth=0)
    leaf_counts = [encrypted_class_counts(ctx, lw, dataset.enc_labels) for lw in leaf_weights]
    return node_splits, leaf_counts
