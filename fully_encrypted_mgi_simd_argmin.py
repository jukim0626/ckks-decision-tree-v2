"""fully_encrypted_mgi_stump.py의 O(n) 순차 tournament argmin을 O(log n) SIMD 기반
divide-and-conquer(reduction)로 교체한 버전.

docs/2024-529.pdf(HBDT)의 FindMaxGroupPos는 4-way 그룹으로 나눠서 log4(n)번만
ApproxSign을 호출하는데, 여기서는 구현 난이도를 낮추기 위해 2-way(절반씩) reduction을
쓴다 (log2(n)번 호출 - log4보다 라운드 수는 많지만, "candidate 하나씩 순서대로 비교"였던
기존 O(n) 방식보다는 여전히 극적으로 적음. n=12면 11번 -> 4번).

핵심 아이디어: score/threshold/counts를 candidate별로 각각 별도 ciphertext에 들고 있던
이전 방식과 달리, "candidate n개의 값"을 SIMD 슬롯 0..n-1에 packing한 ciphertext 하나로
만든다. 그러면 한 라운드의 sign_bootstrap 호출 한 번이 n개 후보 전부의 절반씩을 동시에
비교한다 (기존은 후보 1쌍 비교에 sign_bootstrap 1세트가 필요했음).
"""

from __future__ import annotations

import gc
import time

import numpy as np

from fully_encrypted_mgi_stump import (
    BootstrapTrainingContext,
    create_bootstrap_context,
    encrypted_blend,
    evaluate_candidate,
    ensure_level,
    plaintext_mgi_best_split,
    sharpened_sign,
)
from client_assisted.candidates import (
    PublicSplitCandidate,
    make_public_grid_candidates,
    make_small_public_threshold_grid,
)
from client_assisted.dataset import (
    EncryptedDataset,
    encrypt_dataset,
    load_scaled_dataset_subset,
    one_hot_encode,
)


def next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def scatter_to_slot(ctx, ct, slot_idx: int):
    """scalar 값(모든 슬롯에 broadcast돼 있음 - engine.sum()이 slot 0뿐 아니라 전체
    슬롯에 합계를 복제하기 때문)을 slot_idx 하나에만 남기고 나머지는 0으로 지운 뒤 옮긴다.

    처음에는 "값이 slot 0에만 있다"고 가정하고 그냥 rotate만 했는데, 실제로는
    engine.sum()이 전체 슬롯에 값을 broadcast해서 candidate 10개를 packing하면 전부
    합쳐진 하나의 값이 모든 슬롯에 나오는 버그가 있었다 (모든 candidate 슬롯이 994.9로
    동일하게 나옴 - `evaluate_all_candidates_packed` 첫 디버그에서 발견). 반드시 slot 0
    마스크로 나머지 슬롯을 0으로 지운 뒤 옮겨야 한다.
    """
    mask = np.zeros(ctx.engine.slot_count)
    mask[0] = 1.0
    masked = ctx.engine.multiply(ct, mask)
    if slot_idx == 0:
        return masked
    return ctx.engine.rotate(masked, ctx.rotation_key, slot_idx)


def pack_candidate_scalars(ctx, scalar_cts: list):
    """slot 0에 값 하나씩 들어있는 ciphertext n개를, slot 0..n-1에 각각 값이 들어간
    ciphertext 하나로 packing."""
    packed = None
    for i, ct in enumerate(scalar_cts):
        piece = scatter_to_slot(ctx, ct, i)
        packed = piece if packed is None else ctx.engine.add(packed, piece)
    return packed


def evaluate_all_candidates_packed(
    ctx, dataset: EncryptedDataset, candidates: list, score_normalizer: float, verbose: bool
):
    """candidate 전부의 MGI score/left_counts/right_counts를 계산해서 SIMD packing.

    threshold/feature indicator는 public(공개 candidate 목록에서 바로 계산 가능)이라
    candidate별로 개별 계산할 필요 없이 벡터를 만들어서 바로 encrypt한다.
    """
    n = len(candidates)
    n_pow2 = next_power_of_two(n)
    n_features = dataset.n_features
    n_classes = dataset.n_classes

    # candidate 전부를 계산해서 리스트에 쌓아뒀다가 한꺼번에 packing하면(예전 버전) candidate
    # 수만큼의 ciphertext가 packing 직전까지 전부 GPU에 동시에 살아있어야 해서 candidate가
    # 많을 때(예: digits candidates=192, n_classes=10 -> count ciphertext만 192*10*2=3840개)
    # OOM의 직접적인 원인이 된다 (2026-07-29 digits 벤치마크에서 실제로 재현됨). server_ops.py의
    # candidate 스트리밍(2026-07-07)과 같은 원칙으로, candidate 하나 계산 -> 바로 slot에
    # scatter+누적 -> 즉시 폐기하도록 바꿔서 peak 메모리를 O(n)에서 O(1)로 낮춘다.
    packed_score = None
    packed_left = [None] * n_classes
    packed_right = [None] * n_classes
    for idx, candidate in enumerate(candidates):
        score, lc, rc = evaluate_candidate(ctx, dataset, candidate)
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
            print(f"[simd argmin] candidate {idx + 1}/{n} scored+packed", flush=True)

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


def batch_ensure_level(ctx, ciphertexts: list, min_level: int = 8) -> list:
    """여러 ciphertext 중 level이 낮은 것들을 2개씩 merge_bootstrap으로 묶어서 복구.

    merge_bootstrap(ct1, ct2, ...)은 개별 bootstrap 2번보다 ~2배 빠르면서 값은 완전히
    동일하다 (bench_merge_bootstrap으로 확인). 라운드마다 aux(threshold/feature
    indicator/counts) ciphertext 여러 개를 하나씩 순서대로 bootstrap하던 것을 페어로
    묶어서 절반으로 줄인다. 홀수 개 남으면 마지막 하나만 단독 bootstrap.
    """
    # merge_bootstrap/bootstrap은 입력이 NTT form이면 거부한다("should not be in NTT
    # form") - ct-ct multiply 체인을 여러 단계 거친 ciphertext(특히 depth>1 tree에서
    # 노드를 넘나든 것)는 NTT form일 수 있어서, 호출 전에 무조건 intt로 정규화한다 (intt는
    # 이미 normal form이어도 값이 안 깨지는 안전한 연산 - fully_encrypted_mgi_stump.ensure_level
    # 참고).
    result = [ctx.engine.intt(ct) for ct in ciphertexts]
    needs_refresh = [i for i, ct in enumerate(result) if ct.level < min_level]
    i = 0
    while i < len(needs_refresh):
        if i + 1 < len(needs_refresh):
            idx1, idx2 = needs_refresh[i], needs_refresh[i + 1]
            r1, r2 = ctx.engine.merge_bootstrap(
                result[idx1], result[idx2], ctx.rlk, ctx.conjugation_key, ctx.rotation_key, ctx.small_bootstrap_key
            )
            result[idx1], result[idx2] = ctx.engine.intt(r1), ctx.engine.intt(r2)
            i += 2
        else:
            idx = needs_refresh[i]
            refreshed = ctx.engine.bootstrap(
                result[idx], ctx.rlk, ctx.conjugation_key, ctx.rotation_key, ctx.small_bootstrap_key
            )
            result[idx] = ctx.engine.intt(refreshed)
            i += 1
    return result


def simd_reduce_argmin(
    ctx, packed_score, aux: list, n_pow2: int, score_normalizer: float, sharpen_iterations: int, verbose: bool
):
    """log2(n_pow2)번의 라운드로 packed_score/aux를 절반씩 접어서 최종 winner(slot 0)를 찾는다."""
    score = packed_score
    n_cur = n_pow2
    round_idx = 0
    while n_cur > 1:
        half = n_cur // 2

        refreshed = batch_ensure_level(ctx, [score] + aux)
        score, aux = refreshed[0], refreshed[1:]

        rotated_score = ctx.engine.rotate(score, ctx.rotation_key, -half)
        diff = ctx.engine.subtract(rotated_score, score)  # diff[i] = score[i+half] - score[i]
        normalized = ctx.engine.multiply(diff, 1.0 / score_normalizer)
        sign = sharpened_sign(ctx, normalized, sharpen_iterations)  # +1이면 score[i]가 더 작음(=i가 승)
        choose_first = ctx.engine.multiply(ctx.engine.add(sign, 1.0), 0.5)

        score = encrypted_blend(ctx, choose_first, score, rotated_score)
        new_aux = []
        for a in aux:
            rotated_a = ctx.engine.rotate(a, ctx.rotation_key, -half)
            new_aux.append(encrypted_blend(ctx, choose_first, a, rotated_a))
        aux = new_aux

        n_cur = half
        round_idx += 1
        if verbose:
            print(f"[simd argmin] round {round_idx} done (n={n_cur})", flush=True)
    return score, aux


def train_fully_encrypted_mgi_stump_simd(
    ctx,
    dataset: EncryptedDataset,
    candidates: list,
    score_normalizer: float | None = None,
    sharpen_iterations: int = 20,
    verbose: bool = True,
):
    if score_normalizer is None:
        score_normalizer = float(dataset.n_samples**2)

    packed_score, aux, n_pow2, n_features, n_classes = evaluate_all_candidates_packed(
        ctx, dataset, candidates, score_normalizer, verbose
    )
    winner_score, winner_aux = simd_reduce_argmin(
        ctx, packed_score, aux, n_pow2, score_normalizer, sharpen_iterations, verbose
    )
    winner_threshold = winner_aux[0]
    winner_feature_indicators = winner_aux[1 : 1 + n_features]
    winner_left = winner_aux[1 + n_features : 1 + n_features + n_classes]
    winner_right = winner_aux[1 + n_features + n_classes :]
    return {
        "score": winner_score,
        "threshold": winner_threshold,
        "feature_selectors": winner_feature_indicators,
        "left_counts": winner_left,
        "right_counts": winner_right,
    }


def train_fully_encrypted_mgi_stump_simd_precise_score(
    ctx,
    dataset: EncryptedDataset,
    candidates: list,
    score_normalizer: float | None = None,
    sharpen_iterations: int = 20,
    verbose: bool = True,
):
    """train_fully_encrypted_mgi_stump_simd()와 같은 SIMD reduction으로 승자를 찾되,
    score만 다르게 계산 - "매 라운드 blend되며 반복 bootstrap을 겪은 score"를 그대로
    반환하는 대신, 승자가 확정된 뒤 **한 번도 안 건드린 원본 packed_left/right_counts**에
    승자 위치의 one-hot mask를 곱해서 score를 새로 계산한다.

    가설(EXPERIMENT_LOG.md 2026-07-28 "score 정밀도 저하"): winner.score가 -4.84로 크게
    틀어진 원인이 라운드마다 ensure_level()로 반복 bootstrap된 근사오차 누적이라면, score를
    한 번만(bootstrap 없이) 계산하면 원래 per-candidate score 수준의 정밀도(예: 13.12 vs
    plaintext 0 처럼 작은 오차)로 돌아올 것이다.

    threshold/feature_indicator는 public 정보라 decrypt해서 어느 candidate가 이겼는지
    찾는 데 쓴다 (client-assisted 프로토콜에서도 client가 selected split을 아는 것과
    동일한 수준의 공개 - counts/score 자체는 여전히 encrypted 상태로만 다룬다).
    """
    if score_normalizer is None:
        score_normalizer = float(dataset.n_samples**2)

    packed_score, aux, n_pow2, n_features, n_classes = evaluate_all_candidates_packed(
        ctx, dataset, candidates, score_normalizer, verbose
    )
    # aux 원본(한 번도 blend/bootstrap 안 거친 packed left/right counts)을 따로 보관
    original_left = aux[1 + n_features : 1 + n_features + n_classes]
    original_right = aux[1 + n_features + n_classes :]

    winner_score_blended, winner_aux = simd_reduce_argmin(
        ctx, packed_score, list(aux), n_pow2, score_normalizer, sharpen_iterations, verbose
    )
    winner_threshold_ct = winner_aux[0]
    winner_feature_cts = winner_aux[1 : 1 + n_features]

    # 승자가 어느 candidate인지 찾기 (threshold/feature는 public 값이라 decrypt해서 매칭)
    decrypted_threshold = float(ctx.engine.decrypt(winner_threshold_ct, ctx.sk)[0].real)
    decrypted_feature_conf = [
        float(ctx.engine.decrypt(fc, ctx.sk)[0].real) for fc in winner_feature_cts
    ]
    winner_feature_idx = max(range(n_features), key=lambda i: decrypted_feature_conf[i])

    best_match_idx, best_match_dist = None, None
    for idx, c in enumerate(candidates):
        if c.feature_idx != winner_feature_idx:
            continue
        dist = abs(c.threshold - decrypted_threshold)
        if best_match_dist is None or dist < best_match_dist:
            best_match_dist, best_match_idx = dist, idx

    # 원본(안 건드린) packed counts에 승자 slot만 남기는 one-hot mask를 곱해서 score 재계산
    mask_vec = np.zeros(n_pow2)
    mask_vec[best_match_idx] = 1.0
    mask_ct = ctx.engine.encrypt(mask_vec, ctx.pk)

    def masked_sum(packed_ct):
        masked = ctx.engine.multiply(packed_ct, mask_ct, ctx.rlk)
        return ctx.engine.sum(masked, ctx.rotation_key)

    fresh_left = [masked_sum(c) for c in original_left]
    fresh_right = [masked_sum(c) for c in original_right]
    fresh_score = ctx.engine.add(
        encrypted_mgi_from_counts_local(ctx, fresh_left),
        encrypted_mgi_from_counts_local(ctx, fresh_right),
    )

    return {
        "score_blended": winner_score_blended,
        "score_precise": fresh_score,
        "threshold": winner_threshold_ct,
        "feature_selectors": winner_feature_cts,
        "matched_candidate_idx": best_match_idx,
    }


def encrypted_mgi_from_counts_local(ctx, class_counts: list):
    total = class_counts[0]
    for count in class_counts[1:]:
        total = ctx.engine.add(total, count)
    squared_total = ctx.engine.multiply(total, total, ctx.rlk)
    squared_sum = ctx.engine.multiply(class_counts[0], class_counts[0], ctx.rlk)
    for count in class_counts[1:]:
        squared_sum = ctx.engine.add(squared_sum, ctx.engine.multiply(count, count, ctx.rlk))
    return ctx.engine.subtract(squared_total, squared_sum)


def debug_decrypt_simd_winner(ctx, winner: dict) -> dict:
    return {
        "score": float(ctx.engine.decrypt(winner["score"], ctx.sk)[0].real),
        "threshold": float(ctx.engine.decrypt(winner["threshold"], ctx.sk)[0].real),
        "feature_selectors": [
            float(ctx.engine.decrypt(fs, ctx.sk)[0].real) for fs in winner["feature_selectors"]
        ],
    }


def _smoke_test_synthetic() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    n = 20
    X = np.zeros((n, 2))
    X[:10, 0] = rng.uniform(-1.0, -0.2, size=10)
    X[10:, 0] = rng.uniform(0.2, 1.0, size=10)
    X[:, 1] = rng.uniform(-1.0, 1.0, size=n)
    y = np.array([0] * 10 + [1] * 10)
    y_one_hot = one_hot_encode(y, n_classes=2)
    candidates = make_public_grid_candidates(
        n_features=2, thresholds=make_small_public_threshold_grid(candidate_count=5)
    )

    best_idx, best_score = plaintext_mgi_best_split(X, y_one_hot, candidates)
    print(f"[plaintext] best candidate={candidates[best_idx]} score={best_score:.4f}")

    ctx = create_bootstrap_context(mode="gpu")
    dataset = encrypt_dataset(ctx, X, y_one_hot)

    t0 = time.time()
    winner = train_fully_encrypted_mgi_stump_simd(ctx, dataset, candidates, sharpen_iterations=20)
    elapsed = time.time() - t0

    decoded = debug_decrypt_simd_winner(ctx, winner)
    print(f"\n[simd fully-encrypted] elapsed={elapsed:.2f}s")
    print(f"  score decrypt={decoded['score']:.4f} (plaintext={best_score:.4f})")
    print(f"  threshold decrypt={decoded['threshold']:.4f} (plaintext={candidates[best_idx].threshold:.4f})")
    print(f"  feature_selectors decrypt={[f'{v:.4f}' for v in decoded['feature_selectors']]}")
    print(f"  expected feature_idx={candidates[best_idx].feature_idx}")


if __name__ == "__main__":
    _smoke_test_synthetic()
