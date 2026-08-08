"""batched_argmin_v2의 문제(batch마다 독립적으로 완전한 reduction을 돌려서 총 라운드 수가
log2(n)보다 훨씬 많아짐)를 해결한 버전.

핵심: **순서를 바꾼다.**
- v2: batch 내부 reduction(log2(K)라운드) x n_batches번 먼저 -> 그다음 batch간 결합
      (log2(n_batches)라운드). 총 라운드 = n_batches*log2(K) + log2(n_batches).
- v3: **batch끼리(ciphertext 통째로) 먼저 비교**(log2(n_batches)라운드, rotate 불필요 -
      같은 슬롯 레이아웃인 다른 ciphertext끼리라 그냥 바로 비교) -> 살아남은 ciphertext
      1개에 대해서만 block 내부 reduction(log2(K)라운드). 총 라운드 =
      log2(n_batches) + log2(K) = log2(n_batches*K) ~= log2(n) - 배치 안 했을 때와 거의
      같은 라운드 수를 유지하면서 sigmoid 호출만 n_batches번으로 줄인다.
"""

from __future__ import annotations

import gc

import numpy as np

from batched_candidate_scoring import (
    evaluate_candidate_batch_counts,
    max_batch_size,
    next_pow2,
)
from batched_argmin_v2 import block_strided_reduce
from fully_encrypted_mgi_simd_argmin import batch_ensure_level, encrypted_blend, sharpened_sign


def build_batch_packed_fixed(ctx, dataset, batch_candidates: list, block_size: int, k_pow2: int, score_normalizer: float):
    """batch 하나를 항상 같은 크기(k_pow2 block)로 packing (batch마다 실제 candidate 수가
    달라도 - 특히 마지막 batch - 부족한 자리는 sentinel로 패딩해서 모든 batch의 구조를
    통일한다 - 뒤에서 batch-ciphertext끼리 바로 비교하려면 슬롯 레이아웃이 같아야 함)."""
    k = len(batch_candidates)
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
    return packed_score, aux


def all_sentinel_batch(ctx, dataset, block_size: int, k_pow2: int, score_normalizer: float, n_aux: int):
    """n_batches가 2의 거듭제곱이 아닐 때 채워 넣는 완전 dummy batch(전부 sentinel score,
    절대 안 뽑힘)."""
    slot_count = ctx.engine.slot_count
    pad_vec = np.full(slot_count, score_normalizer)
    packed_score = ctx.engine.encrypt(pad_vec, ctx.pk)
    zero_ct = ctx.engine.encrypt(np.zeros(slot_count), ctx.pk)
    aux = [zero_ct for _ in range(n_aux)]
    return packed_score, aux


def train_fully_encrypted_mgi_stump_batched_v3(
    ctx, dataset, candidates: list, score_normalizer: float | None = None,
    sharpen_iterations: int = 12, verbose: bool = True,
):
    if score_normalizer is None:
        score_normalizer = float(dataset.n_samples**2)

    block_size, max_k = max_batch_size(ctx, dataset.n_samples)
    k_pow2 = next_pow2(max_k)
    n = len(candidates)
    n_batches = (n + max_k - 1) // max_k
    n_batches_pow2 = next_pow2(n_batches)
    n_aux = 1 + dataset.n_features + 2 * dataset.n_classes

    batch_scores = []
    batch_auxs = []
    for b in range(n_batches):
        batch = candidates[b * max_k : (b + 1) * max_k]
        packed_score, aux = build_batch_packed_fixed(ctx, dataset, batch, block_size, k_pow2, score_normalizer)
        batch_scores.append(packed_score)
        batch_auxs.append(aux)
        gc.collect()
        if verbose:
            print(f"[v3 scoring] batch {b + 1}/{n_batches} done ({len(batch)} candidates)", flush=True)

    while len(batch_scores) < n_batches_pow2:
        s, a = all_sentinel_batch(ctx, dataset, block_size, k_pow2, score_normalizer, n_aux)
        batch_scores.append(s)
        batch_auxs.append(a)

    # Step A: batch(ciphertext) 단위로 binary tree 비교 - rotate 불필요(같은 레이아웃끼리
    # 바로 비교), 한 라운드에 sign_bootstrap 1번으로 k_pow2개 block 전부 동시 비교.
    m = n_batches_pow2
    round_idx = 0
    while m > 1:
        half = m // 2
        new_scores, new_auxs = [], []
        for i in range(half):
            a_score, b_score = batch_scores[i], batch_scores[i + half]
            a_aux, b_aux = batch_auxs[i], batch_auxs[i + half]
            refreshed = batch_ensure_level(ctx, [a_score, b_score] + a_aux + b_aux)
            a_score, b_score = refreshed[0], refreshed[1]
            a_aux = refreshed[2 : 2 + n_aux]
            b_aux = refreshed[2 + n_aux :]

            diff = ctx.engine.subtract(b_score, a_score)
            normalized = ctx.engine.multiply(diff, 1.0 / score_normalizer)
            sign = sharpened_sign(ctx, normalized, sharpen_iterations)
            choose_first = ctx.engine.multiply(ctx.engine.add(sign, 1.0), 0.5)

            new_scores.append(encrypted_blend(ctx, choose_first, a_score, b_score))
            new_auxs.append([encrypted_blend(ctx, choose_first, x, y) for x, y in zip(a_aux, b_aux)])
        batch_scores, batch_auxs = new_scores, new_auxs
        m = half
        round_idx += 1
        if verbose:
            print(f"[v3 cross-batch] round {round_idx} done (batches left={m})", flush=True)

    # Step B: 마지막 남은 ciphertext 1개(k_pow2 block)를 block-strided reduction
    final_score, final_aux = block_strided_reduce(
        ctx, batch_scores[0], batch_auxs[0], k_pow2, block_size, score_normalizer, sharpen_iterations, verbose
    )
    return final_score, final_aux
