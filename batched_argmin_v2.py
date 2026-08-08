"""batched_candidate_scoring.py의 block-strided 채점 결과를, 변환(mask+intt+sum) 없이
argmin reduction이 직접 소비하도록 2단계로 재설계.

Level 1 (batch 내부): candidate i의 값이 슬롯 i*block_size에 있는 상태 그대로,
rotate(-half*block_size)로 block 단위 divide-and-conquer - fully_encrypted_mgi_simd_argmin
의 simd_reduce_argmin과 완전히 같은 로직이고 rotate 폭만 block_size배.
Level 2 (batch 사이, batch가 여러 개일 때만): 각 batch 승자(슬롯 0)를 mask+rotate(값
손실 없는 저비용 연산, engine.sum() 안 씀)로 "batch index별 슬롯 1개" 포맷으로 모아서
기존 simd_reduce_argmin(stride=1)을 그대로 재사용해 최종 승자를 찾는다.
"""

from __future__ import annotations

import gc

import numpy as np

from batched_candidate_scoring import (
    evaluate_candidate_batch_counts,
    max_batch_size,
    next_pow2,
)
from fully_encrypted_mgi_simd_argmin import (
    batch_ensure_level,
    encrypted_blend,
    next_power_of_two,
    pack_candidate_scalars,
    scatter_to_slot,
    sharpened_sign,
    simd_reduce_argmin,
)


def scatter_clean_to_slot(ctx, ct, slot_idx: int):
    """이미 slot 0만 값이 있고 나머지가 0인(mask_slot_zero를 거친) ciphertext를 그대로
    rotate만 해서 옮긴다 - fully_encrypted_mgi_simd_argmin.scatter_to_slot처럼 매번
    다시 mask(multiply)를 걸면 NTT form 문제가 재발해서(merge_bootstrap이 거부),
    이미 깨끗한 입력에는 rotate만 적용하는 가벼운 버전을 따로 둔다."""
    if slot_idx == 0:
        return ct
    return ctx.engine.rotate(ct, ctx.rotation_key, slot_idx)


def pack_clean_scalars(ctx, scalar_cts: list):
    """scatter_clean_to_slot 기반 pack_candidate_scalars 대체판 (재-mask 없음)."""
    packed = None
    for i, ct in enumerate(scalar_cts):
        piece = scatter_clean_to_slot(ctx, ct, i)
        packed = piece if packed is None else ctx.engine.add(packed, piece)
    return packed


def mask_slot_zero(ctx, ct):
    """slot 0만 남기고 나머지 0으로 지운다.

    multiply(ciphertext, plaintext array)가 NTT form 결과를 내놓는데, 이후
    merge_bootstrap/bootstrap이 NTT form 입력을 거부해서(engine.sum()에서 겪은 것과
    같은 문제) intt로 되돌려야 한다.
    """
    mask = np.zeros(ctx.engine.slot_count)
    mask[0] = 1.0
    masked = ctx.engine.multiply(ct, mask)
    return ctx.engine.intt(masked)


def build_batch_packed(ctx, dataset, batch_candidates: list, block_size: int, score_normalizer: float):
    """batch 하나(<=max_k candidates)의 score/threshold/feature_indicator/counts를
    block-strided(candidate i -> slot i*block_size)로 packing."""
    k = len(batch_candidates)
    k_pow2 = next_pow2(k)
    n_features = dataset.n_features
    slot_count = ctx.engine.slot_count

    left_counts, right_counts = evaluate_candidate_batch_counts(ctx, dataset, batch_candidates, block_size)

    def _mgi_term(counts):
        total = counts[0]
        for ct in counts[1:]:
            total = ctx.engine.add(total, ct)
        squared_total = ctx.engine.multiply(total, total, ctx.rlk)
        squared_sum = ctx.engine.multiply(counts[0], counts[0], ctx.rlk)
        for ct in counts[1:]:
            squared_sum = ctx.engine.add(squared_sum, ctx.engine.multiply(ct, ct, ctx.rlk))
        return ctx.engine.subtract(squared_total, squared_sum)

    packed_score = ctx.engine.add(_mgi_term(left_counts), _mgi_term(right_counts))

    threshold_vec = np.zeros(slot_count)
    feature_vecs = [np.zeros(slot_count) for _ in range(n_features)]
    for i, c in enumerate(batch_candidates):
        threshold_vec[i * block_size] = c.threshold
        feature_vecs[c.feature_idx][i * block_size] = 1.0
    packed_threshold = ctx.engine.encrypt(threshold_vec, ctx.pk)
    packed_features = [ctx.engine.encrypt(v, ctx.pk) for v in feature_vecs]

    if k_pow2 > k:
        pad_vec = np.zeros(slot_count)
        for i in range(k, k_pow2):
            pad_vec[i * block_size] = score_normalizer
        packed_score = ctx.engine.add(packed_score, ctx.engine.encrypt(pad_vec, ctx.pk))

    aux = [packed_threshold] + packed_features + left_counts + right_counts
    return packed_score, aux, k_pow2


def block_strided_reduce(
    ctx, packed_score, aux: list, k_pow2: int, block_size: int, score_normalizer: float,
    sharpen_iterations: int, verbose: bool,
):
    """simd_reduce_argmin과 동일 로직, rotate 폭만 half -> half*block_size."""
    score = packed_score
    n_cur = k_pow2
    round_idx = 0
    while n_cur > 1:
        half = n_cur // 2
        refreshed = batch_ensure_level(ctx, [score] + aux)
        score, aux = refreshed[0], refreshed[1:]

        rotated_score = ctx.engine.rotate(score, ctx.rotation_key, -half * block_size)
        diff = ctx.engine.subtract(rotated_score, score)
        normalized = ctx.engine.multiply(diff, 1.0 / score_normalizer)
        sign = sharpened_sign(ctx, normalized, sharpen_iterations)
        choose_first = ctx.engine.multiply(ctx.engine.add(sign, 1.0), 0.5)

        score = encrypted_blend(ctx, choose_first, score, rotated_score)
        new_aux = []
        for a in aux:
            rotated_a = ctx.engine.rotate(a, ctx.rotation_key, -half * block_size)
            new_aux.append(encrypted_blend(ctx, choose_first, a, rotated_a))
        aux = new_aux

        n_cur = half
        round_idx += 1
        if verbose:
            print(f"[block-strided reduce] round {round_idx} done (n={n_cur})", flush=True)
    return score, aux


def train_fully_encrypted_mgi_stump_batched(
    ctx, dataset, candidates: list, score_normalizer: float | None = None,
    sharpen_iterations: int = 12, verbose: bool = True,
):
    """batched scoring(sigmoid 호출 수 최소화) + block-strided argmin(변환 비용 없음)을
    합친 2단계 파이프라인. 반환 포맷은 simd_reduce_argmin과 호환(score, aux)."""
    if score_normalizer is None:
        score_normalizer = float(dataset.n_samples**2)

    block_size, max_k = max_batch_size(ctx, dataset.n_samples)
    n = len(candidates)
    n_batches = (n + max_k - 1) // max_k

    batch_winners_score = []
    batch_winners_aux = []
    for b in range(n_batches):
        batch = candidates[b * max_k : (b + 1) * max_k]
        packed_score, aux, k_pow2 = build_batch_packed(ctx, dataset, batch, block_size, score_normalizer)
        winner_score, winner_aux = block_strided_reduce(
            ctx, packed_score, aux, k_pow2, block_size, score_normalizer, sharpen_iterations, verbose
        )
        batch_winners_score.append(mask_slot_zero(ctx, winner_score))
        batch_winners_aux.append([mask_slot_zero(ctx, a) for a in winner_aux])
        gc.collect()
        if verbose:
            print(f"[batched argmin] batch {b + 1}/{n_batches} reduced ({len(batch)} candidates)", flush=True)

    if n_batches == 1:
        return batch_winners_score[0], batch_winners_aux[0]

    # level 2: batch 승자들(슬롯 0)을 batch index별 슬롯으로 모아서 기존 simd_reduce_argmin 재사용
    n_batches_pow2 = next_power_of_two(n_batches)
    packed_score2 = pack_clean_scalars(ctx, batch_winners_score)
    if n_batches_pow2 > n_batches:
        pad_vec = np.array([0.0] * n_batches + [score_normalizer] * (n_batches_pow2 - n_batches))
        packed_score2 = ctx.engine.add(packed_score2, ctx.engine.encrypt(pad_vec, ctx.pk))

    n_aux = len(batch_winners_aux[0])
    aux2 = [pack_clean_scalars(ctx, [batch_winners_aux[b][a_idx] for b in range(n_batches)]) for a_idx in range(n_aux)]

    return simd_reduce_argmin(ctx, packed_score2, aux2, n_batches_pow2, score_normalizer, sharpen_iterations, verbose)
