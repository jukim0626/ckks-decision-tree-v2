"""Encrypted packed softmax (일반 K-way softmax, MGI 특화 로직 없음) - exp(beta*x)/sum을
bounded Newton-Raphson reciprocal로 나눗셈 없이 계산. gradient_soft_tree의 leaf
classifier/attention뿐 아니라 local_loss의 레벨별 local classifier에도 재사용된다.

2026-09-09 리팩터로 experiments/gradient_soft_tree/depth1_ckks.py에서 분리됨(원래 이
파일 상단에 정의돼 있었음 - depth1 전용 학습 로직과 뒤섞여 있었으나 실제로는 depth 무관한
범용 프리미티브였음)."""

from __future__ import annotations

import gc

import numpy as np

from core.ckks_engine import ensure_level
from core.approximation.exp import chebyshev_approximation_exp

SOFTMAX_EXP_DEGREE = 20
SOFTMAX_EXP_INTERVAL = (-2.5, 2.5)
# 2026-08-26: soft_mgi.py의 z-score reciprocal은 22회가 필요했지만, 여기서는 z0=mean(rescaled
# exp_val)이 항상 (0,1] 안이고 alpha/leaf_logits 값 자체가 크게 안 벌어지므로(plaintext 실측:
# 학습 내내 |alpha|,|leaf_logits| < ~2) z0가 0.05 밑으로 내려가는 경우가 거의 없다. bounded NR은
# e_{k+1}=e_k^2 이차수렴이라, 최악 z0=0.05(e0=0.95)에서도 k=8이면 e_k~1e-4로 이미 충분 - 10으로
# 안전마진을 조금 더 두고 씀.
SOFTMAX_RECIP_ITERATIONS = 10


def softmax_exp_coeffs(interval: tuple[float, float] = SOFTMAX_EXP_INTERVAL, degree: int = SOFTMAX_EXP_DEGREE) -> list[float]:
    """exp(z)(z in interval)의 Chebyshev 계수. exp(-beta*x) 피팅 함수에 beta=-1을 줘서
    exp(+x)를 얻는다."""
    return chebyshev_approximation_exp(degree, beta=-1.0, interval=interval).tolist()


_EXP_COEFFS = softmax_exp_coeffs()
_EXP_MAX = float(np.exp(SOFTMAX_EXP_INTERVAL[1]))  # 공개 상수: exp(2.5), rescale로 (0,1] 안에 가둠


def packed_softmax(
    ctx, values_packed, n_valid: int, n_pow2: int,
    reciprocal_iterations: int = SOFTMAX_RECIP_ITERATIONS, min_level: int = 5,
):
    """values_packed(슬롯 0..n_valid-1에 값, 나머지 패딩 0) -> 합이 1인 softmax 벡터(같은 레이아웃).

    exp_val을 공개 상한 exp(2.5)로 미리 나눠서 (0,1] 안에 가둬 두는 트릭으로
    z0=mean(exp_val)이 항상 <=1이 되게 만들어 bounded reciprocal이 안전하게 수렴하게 한다.
    """
    valid_mask = np.array([1.0] * n_valid + [0.0] * (n_pow2 - n_valid))
    z = ctx.engine.multiply(values_packed, valid_mask)
    z = ctx.engine.intt(z)
    # degree=20 exp poly가 레벨을 ~6 소모하고 뒤이은 스칼라 곱 2번이 ~2 더 소모(합 ~8) -
    # poly 진입 전에 미리 높은 min_level로 부트스트랩해서 "input ciphertext should have
    # a positive level" 에러를 방지한다.
    z = ensure_level(ctx, z, min_level=10)
    exp_val = ctx.engine.evaluate_polynomial(z, _EXP_COEFFS, ctx.rlk)
    exp_val = ctx.engine.multiply(exp_val, 1.0 / _EXP_MAX)
    exp_val = ctx.engine.multiply(exp_val, valid_mask)  # exp(0)=1 패딩 재오염 방지

    exp_val_for_sum = ctx.engine.intt(exp_val)
    denom = ctx.engine.sum(exp_val_for_sum, ctx.rotation_key)
    denom = ensure_level(ctx, denom, min_level=min_level)
    exp_val = ensure_level(ctx, exp_val, min_level=min_level)

    y0 = 1.0 / n_valid
    z_iter = ctx.engine.multiply(denom, y0)
    w = ctx.engine.multiply(exp_val, y0)
    for i in range(reciprocal_iterations):
        z_iter = ensure_level(ctx, z_iter, min_level=min_level)
        w = ensure_level(ctx, w, min_level=min_level)
        two_minus_z = ctx.engine.subtract(2.0, z_iter)
        z_new = ctx.engine.multiply(z_iter, two_minus_z, ctx.rlk)
        w = ctx.engine.multiply(w, two_minus_z, ctx.rlk)
        z_iter = z_new
        del two_minus_z
        if i % 5 == 0:
            gc.collect()
    return w


def softmax_backward_packed(ctx, dist_packed, dL_ddist_packed, n_valid: int, n_pow2: int, min_level: int = 5):
    """packed dist/dL_ddist(둘 다 슬롯 0..n_valid-1)에 대해 dL/dlogit = dist*(dL_ddist - dot)
    (dot = sum_c dL_ddist_c*dist_c, 전체 슬롯에 broadcast). 패딩은 dist가 이미 0이라 자동으로 0."""
    dist_packed = ensure_level(ctx, dist_packed, min_level=min_level)
    dL_ddist_packed = ensure_level(ctx, dL_ddist_packed, min_level=min_level)
    prod = ctx.engine.multiply(dist_packed, dL_ddist_packed, ctx.rlk)
    prod = ctx.engine.intt(prod)
    dot = ensure_level(ctx, ctx.engine.sum(prod, ctx.rotation_key), min_level=min_level)
    diff = ensure_level(ctx, ctx.engine.subtract(dL_ddist_packed, dot), min_level=min_level)
    return ctx.engine.multiply(dist_packed, diff, ctx.rlk)
