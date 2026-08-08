"""여러 candidate의 sigmoid soft-split 계산을 한 ciphertext에 block으로 packing해서
sigmoid_approx_enc 호출 횟수 자체를 줄이는 batched scoring.

지금까지(fully_encrypted_mgi_simd_argmin.py)는 candidate마다 순차적으로 sigmoid를
호출했다 (n_candidates번). Task 2(curve fitting)에서 이 "채점(scoring)" 단계가 SIMD
argmin을 아무리 빠르게 해도 남는 O(n) 병목이라는 게 드러났다 - 이 파일은 그 다음 최적화
시도다.

핵심 아이디어: block_size(= n_samples를 넘는 2의 거듭제곱) 슬롯짜리 block을 K개
이어붙인 ciphertext 하나에, block i마다 (feature_i - threshold_i)를 넣고
sigmoid_approx_enc를 K개 block에 대해 **한 번**만 호출한다. block_size*K <= slot_count
제약이 있어서 K는 n_samples가 작을수록(=block_size가 작을수록) 커진다.
"""

from __future__ import annotations

import time

import numpy as np

from ckks_tree import sigmoid_approx_enc


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def scatter_block(ctx, ct, block_idx: int, block_size: int):
    """ct(길이 <= block_size, 나머지 슬롯은 0)를 block_idx번째 block 위치로 이동."""
    offset = block_idx * block_size
    if offset == 0:
        return ct
    return ctx.engine.rotate(ct, ctx.rotation_key, offset)


def segmented_sum(ctx, ct, block_size: int):
    """block_size 슬롯짜리 block마다 내부적으로 합산 (block 0번 슬롯에 그 block의 합이 남음).

    block_size가 2의 거듭제곱이어야 block 경계를 안 넘어간다 (bench_segmented_sum으로 검증).
    """
    x = ct
    shift = 1
    while shift < block_size:
        rotated = ctx.engine.rotate(x, ctx.rotation_key, -shift)
        x = ctx.engine.add(x, rotated)
        shift *= 2
    return x


def max_batch_size(ctx, n_samples: int) -> tuple[int, int]:
    """slot_count 안에 들어가는 block_size와 한 번에 처리 가능한 candidate 수(K)."""
    block_size = next_pow2(n_samples)
    k = ctx.engine.slot_count // block_size
    return block_size, max(1, k)


def evaluate_candidate_batch(ctx, dataset, candidates_batch: list, block_size: int):
    """같은 batch 안 candidate들(서로 다른 feature/threshold 가능)의 MGI 재료를
    sigmoid_approx_enc **1번**으로 한꺼번에 계산.

    반환: (packed_right_prob, packed_left_prob) - block i(슬롯 i*block_size 기준)에 그
    candidate의 soft-split 확률이 담겨 있다 (segmented_sum 등으로 후처리해서 class count
    계산에 사용).
    """
    k = len(candidates_batch)
    packed_feature = None
    threshold_vec = np.zeros(k * block_size)
    for i, candidate in enumerate(candidates_batch):
        enc_feature = dataset.enc_features[candidate.feature_idx]
        piece = scatter_block(ctx, enc_feature, i, block_size)
        packed_feature = piece if packed_feature is None else ctx.engine.add(packed_feature, piece)
        threshold_vec[i * block_size : (i + 1) * block_size] = candidate.threshold

    diff = ctx.engine.subtract(packed_feature, threshold_vec)
    right_prob = sigmoid_approx_enc(ctx.engine, ctx.rlk, diff)
    left_prob = ctx.engine.subtract(1.0, right_prob)
    return right_prob, left_prob


def evaluate_candidate_batch_counts(ctx, dataset, candidates_batch: list, block_size: int):
    """batch 안 candidate들의 (block-strided) left/right class count와 MGI score를 계산.

    반환 ciphertext들은 block i(슬롯 i*block_size)에 candidate i의 값이 들어있다 - 다른
    슬롯은 segmented_sum 중간 결과(garbage)라 안 쓴다.
    """
    right_prob, left_prob = evaluate_candidate_batch(ctx, dataset, candidates_batch, block_size)
    k = len(candidates_batch)

    packed_label = [None] * dataset.n_classes
    for c in range(dataset.n_classes):
        packed = None
        for i in range(k):
            piece = scatter_block(ctx, dataset.enc_labels[c], i, block_size)
            packed = piece if packed is None else ctx.engine.add(packed, piece)
        packed_label[c] = packed

    left_counts, right_counts = [], []
    for c in range(dataset.n_classes):
        weighted_left = ctx.engine.multiply(left_prob, packed_label[c], ctx.rlk)
        weighted_right = ctx.engine.multiply(right_prob, packed_label[c], ctx.rlk)
        left_counts.append(segmented_sum(ctx, weighted_left, block_size))
        right_counts.append(segmented_sum(ctx, weighted_right, block_size))

    return left_counts, right_counts


def extract_block_value(ctx, ct, block_idx: int, block_size: int) -> float:
    """디버그용: block_idx번째 block의 0번 슬롯(=segmented_sum 결과) 값을 decrypt."""
    return float(ctx.engine.decrypt(ct, ctx.sk)[block_idx * block_size].real)


def scatter_to_global_slot(ctx, ct, block_idx: int, block_size: int, global_idx: int, slot_count: int):
    """batch 안 block_idx(슬롯 block_idx*block_size)에 있는 scalar 값을, 다른 candidate들과
    packing할 전역 slot(global_idx)로 옮긴다 (fully_encrypted_mgi_simd_argmin.py의
    pack_candidate_scalars와 호환되는 포맷).

    slot block_idx*block_size 자리만 마스크로 남기고 -> sum()으로 전체 슬롯에 broadcast
    -> global_idx로 rotate.
    """
    mask = np.zeros(slot_count)
    mask[block_idx * block_size] = 1.0
    masked = ctx.engine.multiply(ct, mask)
    masked = ctx.engine.intt(masked)  # engine.sum()이 NTT form 입력을 거부해서 변환 필요
    broadcast = ctx.engine.sum(masked, ctx.rotation_key)
    if global_idx == 0:
        return broadcast
    return ctx.engine.rotate(broadcast, ctx.rotation_key, global_idx)


def evaluate_all_candidates_packed_batched(
    ctx, dataset, candidates: list, score_normalizer: float, verbose: bool = False
):
    """evaluate_all_candidates_packed()와 같은 출력 포맷(전역 candidate index i가 슬롯 i에
    있는 packed_score/aux)을 만들되, candidate를 하나씩이 아니라 block_size*K<=slot_count
    만큼 batch로 묶어서 sigmoid_approx_enc를 ceil(n/K)번만 호출한다.
    """
    import gc

    from fully_encrypted_mgi_simd_argmin import next_power_of_two

    n = len(candidates)
    n_pow2 = next_power_of_two(n)
    n_features = dataset.n_features
    n_classes = dataset.n_classes
    n_samples = dataset.n_samples

    block_size, max_k = max_batch_size(ctx, n_samples)
    slot_count = ctx.engine.slot_count

    packed_score = None
    packed_left = [None] * n_classes
    packed_right = [None] * n_classes

    n_batches = (n + max_k - 1) // max_k
    for batch_idx in range(n_batches):
        batch = candidates[batch_idx * max_k : (batch_idx + 1) * max_k]
        left_counts, right_counts = evaluate_candidate_batch_counts(ctx, dataset, batch, block_size)

        for local_i, candidate in enumerate(batch):
            global_i = batch_idx * max_k + local_i
            lc_global = [
                scatter_to_global_slot(ctx, ct, local_i, block_size, global_i, slot_count)
                for ct in left_counts
            ]
            rc_global = [
                scatter_to_global_slot(ctx, ct, local_i, block_size, global_i, slot_count)
                for ct in right_counts
            ]

            def _mgi_term(counts):
                total = counts[0]
                for ct in counts[1:]:
                    total = ctx.engine.add(total, ct)
                squared_total = ctx.engine.multiply(total, total, ctx.rlk)
                squared_sum = ctx.engine.multiply(counts[0], counts[0], ctx.rlk)
                for ct in counts[1:]:
                    squared_sum = ctx.engine.add(squared_sum, ctx.engine.multiply(ct, ct, ctx.rlk))
                return ctx.engine.subtract(squared_total, squared_sum)

            candidate_score = ctx.engine.add(_mgi_term(lc_global), _mgi_term(rc_global))
            packed_score = candidate_score if packed_score is None else ctx.engine.add(packed_score, candidate_score)
            for c in range(n_classes):
                packed_left[c] = lc_global[c] if packed_left[c] is None else ctx.engine.add(packed_left[c], lc_global[c])
                packed_right[c] = rc_global[c] if packed_right[c] is None else ctx.engine.add(packed_right[c], rc_global[c])

        del left_counts, right_counts
        gc.collect()
        if verbose:
            print(f"[batched scoring] batch {batch_idx + 1}/{n_batches} done ({len(batch)} candidates)", flush=True)

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
