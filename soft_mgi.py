"""Soft-MGI: fully-encrypted MGI 기반 depth=1 stump의 SIMD tournament argmin
(fully_encrypted_mgi_simd_argmin.simd_reduce_argmin, sign_bootstrap 기반)을 대체하는
대안 경로.

기존 경로: candidate K개의 encrypted MGI 점수 -> log2(K)라운드 SIMD tournament(매 라운드
sign_bootstrap)로 승자 1명만 골라서 그 candidate의 sigmoid gate만 씀.
Soft-MGI: argmin 자체를 없애고, K개의 MGI 점수를 softmax 가중치로 바꿔서 K개의 sigmoid
gate를 가중 평균으로 blend. sign_bootstrap을 한 번도 안 쓴다 (run_soft_mgi_stump.py에서
호출 횟수 0회를 직접 확인).

이 파일은 두 함수만 제공한다:
- soft_mgi_weights: K개 encrypted MGI 점수 -> 합이 1인 encrypted softmax weight 벡터
- blended_gate: weights + 기존 sigmoid gate(재구현 없이 재사용) -> blended routing 값

**scope 밖**: weight로 가중한 자식 노드의 class count(=leaf 분포)를 다시 MGI 공식에 넣는
"weighted-MGI 재유도"는 여기서 다루지 않는다 - class count가 정수가 아니라 실수 weight
합이 되면서 division-free MGI 공식(|S|^2 - sum|S_c|^2)의 전제가 깨지는 문제인데, depth=1은
root 한 번만 MGI를 계산하고 더 이상 자식 노드에서 MGI를 재계산하지 않으므로 해당 없음.
depth>=2로 확장할 때 다시 검토해야 한다.
"""

from __future__ import annotations

import numpy as np

from ckks_tree import sigmoid_approx_enc
from exp_approx_coeffs import exp_neg_beta_coeffs
from fully_encrypted_mgi_stump import ensure_level


def soft_mgi_weights(
    ctx,
    packed_score,
    n_valid: int,
    n_pow2: int,
    score_normalizer: float,
    beta: float,
    degree: int = 20,
    reciprocal_iterations: int = 40,
):
    """packed_score(fully_encrypted_mgi_simd_argmin.evaluate_all_candidates_packed와 동일
    포맷 - 슬롯 0..n_valid-1에 후보 K=n_valid개의 encrypted MGI 점수, n_valid..n_pow2-1은
    패딩)를 softmax 가중치 벡터로 변환.

    반환: 합이 1인(패딩 슬롯은 0으로 마스킹) encrypted weight 벡터. 슬롯 i = candidate i의
    가중치.

    **2026-08-05 디버깅 히스토리 (중요, 다시 실수하지 않도록 기록)**: 처음엔
    "beta=20부터 sum(weights)!=1로 무너지는 게 exp(-beta*x) 다항식 차수(degree=15) 부족
    때문"이라고 진단해서 degree를 20으로 올렸는데 효과가 없었다(beta=20 sum 0.880->0.880
    그대로, beta=30/50은 오히려 weight가 수만 단위로 더 심하게 발산). exp_val을 decrypt해서
    평문과 비교해보니 exp 다항식/denom 합산 전부 1e-7 수준으로 정확했고, 그 다음엔
    "Newton-Raphson 반복을 80회로 너무 넉넉하게 잡아서 수렴 후에도 도는 반복이 noise만
    쌓는다"고 진단해서 35회로 줄였는데 이번엔 **더 나빠졌다**(beta=20 sum 0.880->0.654) -
    이 가설도 틀렸다는 뜻이었다. 반복 단위로 level/decrypt를 전부 추적해서 찾은 진짜 원인은
    완전히 달랐다: **`engine.bootstrap()`은 ciphertext에 담긴 값의 크기가 대략 2~5를 넘으면
    에러 없이 조용히 값을 깨뜨린다** (독립적으로 측정: 값 0.5->오차 0.02%, 10->15%,
    22.8->65%, 50->120%(부호까지 뒤집힘) - `sign_bootstrap`이 입력을 [-1,1]로 정규화해야
    하는 것과 같은 종류의 제약이 일반 `bootstrap()`에도 있다). 이전 구현은 `y=1/D`를 직접
    반복 계산했는데, D가 작을수록(beta가 클수록) y가 수십~수만까지 커지고, 그 상태에서
    level이 떨어져 bootstrap이 걸리면 값이 깨졌다 (그래서 iterations를 몇 회로 조정하든
    "언젠가 큰 값에서 bootstrap이 걸리는" 문제 자체는 안 없어졌던 것).

    **수정**: y를 직접 추적하는 대신, 항상 (0,1) 안에 갇혀있는 두 값만 추적한다.
    - z = D*y (수렴하면 1, 자기완결적 재귀 z<-z*(2-z))
    - w = exp_val*y (우리가 원하는 최종 weight 자체 - softmax 확률이라 원래 [0,1] 안이고,
      0에서 단조증가하며 그 값으로 수렴하므로 도중에도 1을 절대 못 넘음)
    두 값 다 이론적으로 bootstrap의 안전 범위를 벗어나지 않아서, 몇 번을 반복하든(테스트로
    30회 확인, level이 24->7까지 떨어지며 여러 번 bootstrap을 거쳤어도) 더 이상 깨지지 않는다
    (bounded_recursion_test.log, 2026-08-05). `y=1/D` 자체는 이제 계산하지 않으므로
    reciprocal_approx.encrypted_reciprocal은 이 함수에서 더 이상 쓰지 않는다 (그 파일은
    "정말 1/x 자체가 필요한" 다른 용도가 생기면 참고용으로 남겨둠 - 단, 같은 크기 제약을
    반드시 고려해야 함).
    """
    # 1) 정규화: MGI score의 공개 이론적 상한 score_normalizer(=n_samples^2)로 나눠서
    #    [0,1] 안으로 넣는다 (exp_approx_coeffs.py 상단 설명 참고 - 실제 min/max를 찾는
    #    MinMaxScaler 방식은 ciphertext 비교가 필요해서 sign_bootstrap-free 목표와 충돌).
    normalized_score = ctx.engine.multiply(packed_score, 1.0 / score_normalizer)

    # 2) exp(-beta * normalized_score)를 Chebyshev 다항식 근사로 계산 (score가 작을수록,
    #    즉 MGI 기준 더 좋은 candidate일수록 가중치가 커짐 - argmin과 같은 방향).
    exp_coeffs = exp_neg_beta_coeffs(beta, degree=degree)
    exp_val = ctx.engine.evaluate_polynomial(normalized_score, exp_coeffs, ctx.rlk)

    # 3) 패딩 슬롯 마스킹: evaluate_all_candidates_packed()가 패딩 슬롯을
    #    normalized_score=1(최악)로 채워서 exp(-beta*1)이 정확히 0은 아니다 (beta가 작을 때
    #    특히 실제 후보와 비슷한 크기가 돼서 denominator를 오염시킬 수 있음) - 공개 valid
    #    mask(패딩 위치는 원래 후보 목록 길이만 알면 되는 public 정보)로 명시적으로 0 처리.
    #
    #    **버그(2026-08-06 발견)**: 원래 `if n_pow2 > n_valid:`일 때만 마스킹했는데,
    #    n_valid가 이미 2의 거듭제곱이면(예: feature 4개로 candidate를 만든 closed_form_mgi.py
    #    - n_pow2==n_valid) 이 조건이 거짓이 되어 마스킹이 통째로 스킵된다. packed_score는
    #    n_valid개 슬롯 이후(n_valid..engine.slot_count-1, 보통 수만 개)가 전부 0이고,
    #    exp(-beta*0)=1이라 이 "빈" 슬롯들이 전부 1로 채워진 채 sum()에 그대로 들어가서
    #    denom이 수만 배 부풀려지고 이후 Newton-Raphson이 완전히 발산한다(iris feature=4개,
    #    n_pow2=4 케이스로 실제 재현 - denom이 기대값 0.33 대신 32764가 나옴, slot_count와
    #    거의 일치). candidates 개수가 2의 거듭제곱이 아니었던 이전 실험(K=12 등)에서는
    #    n_pow2>n_valid가 항상 참이라 이 버그가 가려져 있었다. 항상 마스킹하도록 조건을
    #    없앤다 - n_valid==n_pow2(패딩 0개)여도 n_pow2 이후 슬롯은 여전히 마스킹해야 한다.
    valid_mask = np.array([1.0] * n_valid + [0.0] * (n_pow2 - n_valid))
    exp_val = ctx.engine.multiply(exp_val, valid_mask)

    # 4) 분모(SIMD 합) - engine.sum()은 NTT form 입력을 거부하므로 intt로 정규화 후 호출
    #    (fully_encrypted_mgi_simd_argmin.scatter_to_global_slot과 동일한 이유).
    exp_val_for_sum = ctx.engine.intt(exp_val)
    denom = ctx.engine.sum(exp_val_for_sum, ctx.rotation_key)  # 전체 슬롯에 합이 broadcast

    # 5) bounded Newton-Raphson: z=D*y, w=exp_val*y를 y0=1/n_valid에서 출발해서
    #    z<-z*(2-z), w<-w*(2-z)로 갱신 (둘 다 (0,1) 안에 갇혀있어 bootstrap-safe).
    y0 = 1.0 / n_valid
    z = ctx.engine.multiply(denom, y0)
    w = ctx.engine.multiply(exp_val, y0)
    for _ in range(reciprocal_iterations):
        z = ensure_level(ctx, z)
        w = ensure_level(ctx, w)
        two_minus_z = ctx.engine.subtract(2.0, z)
        z_new = ctx.engine.multiply(z, two_minus_z, ctx.rlk)
        w = ctx.engine.multiply(w, two_minus_z, ctx.rlk)
        z = z_new

    return w


def _extract_weight_broadcast(ctx, weights, candidate_idx: int):
    """weights의 슬롯 candidate_idx에 있는 값만 남기고 전체 슬롯에 broadcast (다른
    슬롯 레이아웃 - 예: 샘플별 gate ciphertext - 과 곱하려면 스칼라처럼 모든 슬롯에
    같은 값이 있어야 함). fully_encrypted_mgi_simd_argmin.scatter_to_global_slot의
    역방향 연산과 같은 마스크->sum 패턴."""
    mask = np.zeros(ctx.engine.slot_count)
    mask[candidate_idx] = 1.0
    isolated = ctx.engine.multiply(weights, mask)
    isolated = ctx.engine.intt(isolated)
    return ctx.engine.sum(isolated, ctx.rotation_key)


def blended_gate(ctx, enc_features: list, candidates: list, weights):
    """각 후보의 기존 sigmoid gate(sigmoid_approx_enc, 재구현 없이 재사용)를 계산해서
    g(x) = sum_i weights[i] * g_i(x) 를 SIMD 곱셈+합으로 만든다.

    enc_features: dataset.enc_features와 동일 포맷 (feature_idx -> 샘플들이 slot packing된
    ciphertext 하나). 반환값도 같은 샘플-슬롯 레이아웃의 ciphertext(각 슬롯이 그 샘플의
    routing 확률, 0~1) 하나.
    """
    gate = None
    for i, candidate in enumerate(candidates):
        enc_feature = enc_features[candidate.feature_idx]
        enc_diff = ctx.engine.subtract(enc_feature, candidate.threshold)
        g_i = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff)  # 기존 gate 함수 그대로 재사용

        w_i_broadcast = _extract_weight_broadcast(ctx, weights, i)
        piece = ctx.engine.multiply(w_i_broadcast, g_i, ctx.rlk)
        gate = piece if gate is None else ctx.engine.add(gate, piece)
    return gate
