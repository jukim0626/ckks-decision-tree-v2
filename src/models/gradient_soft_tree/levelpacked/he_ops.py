"""levelpacked가 쓰는 CKKS 연산 helper. 전부 `ctx.engine`(desilofhe Engine 또는 numpy 시뮬레이터
`sim_engine.SimEngine`)만 호출하므로 같은 코드가 GPU CKKS와 CPU numpy 양쪽에서 돈다.

ROADMAP 2-4(bootstrap 위치): level 소모가 큰 연산(다항식 평가, Newton 반복, 곱셈) 직전에만
`ensure_level`을 부르고, 회전/덧셈(레벨 소모 없음)에는 부르지 않는다. 처리하는 ciphertext 수 자체가
level당 몇 개뿐이라 "하나를 여러 조각으로 꺼낸 뒤 조각마다 bootstrap"하던 기존 문제도 구조적으로 없다.
"""

from __future__ import annotations

import numpy as np

from core.ckks_engine import ensure_level, ensure_levels

MIN_LEVEL = 4          # 일반 곱셈 직전 최소 레벨
SIGMOID_MIN_LEVEL = 12  # 15차 sigmoid 다항식 직전(기존 packed와 동일)
EXP_MIN_LEVEL = 10      # 20차 exp 다항식 직전(기존 softmax와 동일)


def lv(ctx, ct, min_level: int = MIN_LEVEL):
    return ensure_level(ctx, ct, min_level=min_level)


def lvs(ctx, cts: list, min_level: int = MIN_LEVEL) -> list:
    """여러 ct를 한꺼번에 레벨 확보 - bootstrap이 필요한 것끼리 2개씩 merge_bootstrap(2026-10-09)."""
    return ensure_levels(ctx, cts, min_level=min_level)


def mul(ctx, a, b):
    """ct x ct 또는 ct x 평문(벡터/스칼라). 곱하기 전에 레벨을 확인한다."""
    if isinstance(b, (np.ndarray, float, int)):
        return ctx.engine.multiply(lv(ctx, a), b)
    a, b = lvs(ctx, [a, b])   # 둘 다 bootstrap이 필요하면 한 번에(merge)
    return ctx.engine.multiply(a, b, ctx.rlk)


def rot(ctx, ct, shift: int):
    """rotate(ct, shift)[i] = ct[i - shift] (+shift면 값이 높은 인덱스로 이동). 곱셈 결과는 NTT 형태일 수
    있어서 먼저 일반 형태로 바꾼다(intt는 이미 일반 형태여도 값이 안 바뀜 - core.ckks_engine.ensure_level 참고)."""
    if shift % ctx.engine.slot_count == 0:
        return ct
    return ctx.engine.rotate(ctx.engine.intt(ct), ctx.rotation_key, shift)


def poly(ctx, ct, coeffs, min_level: int):
    """다항식 평가. desilofhe는 NTT 형태 입력을 거부하므로 intt 후 레벨 확보."""
    ct = lv(ctx, ctx.engine.intt(ct), min_level)
    return ctx.engine.evaluate_polynomial(ct, list(coeffs), ctx.rlk)


def fold(ctx, ct, stride: int, count: int):
    """segment(= stride*count 슬롯)의 첫 block에 count개 block의 합을 모은다(첫 block만 유효, 나머지는
    부분합 쓰레기 - 호출부가 마스킹해야 한다). count는 2의 거듭제곱."""
    s = stride
    while s < stride * count:
        ct = ctx.engine.add(ct, rot(ctx, ct, -s))
        s *= 2
    return ct


def spread(ctx, ct, stride: int, count: int):
    """각 segment의 첫 block에만 값이 있다고 보고, 그 값을 count개 block에 복제한다(fold의 역방향).
    첫 block 외엔 0이어야 정확하다."""
    s = stride
    while s < stride * count:
        ct = ctx.engine.add(ct, rot(ctx, ct, s))
        s *= 2
    return ct


def block_allreduce(ctx, ct, lay):
    """block(B 슬롯)마다 합을 구해 그 block 전체에 복제. ct는 block 안의 유효 샘플 밖이 0이어야 한다."""
    ct = fold(ctx, ct, 1, lay.B)
    ct = mul(ctx, ct, lay.block_start_mask())
    return spread(ctx, ct, 1, lay.B)


def seg_softmax(ctx, zs: list, lay, valid_counts: list, count: int, exp_coeffs, exp_interval: float, recip_iterations: int) -> list:
    """segment(노드)마다 block 축(feature 또는 class)으로 softmax. 축이 여러 ciphertext(feature 그룹)에 나뉘어
    있으면 zs/valid_counts에 그룹별로 넘긴다 - 분모만 그룹 합을 더해서 공유한다.
    `core.encrypted_ops.softmax.packed_softmax`와 같은 계산(exp 다항식 / exp(구간 상한), Newton 역수, 초기값
    1/전체 유효 개수) - reference.PolySpec 에뮬레이션과 수식이 같다."""
    n_valid = sum(valid_counts)
    es, denom = [], None
    for z, nv in zip(zs, valid_counts):
        valid = lay._mask(count, nv, lay.B)
        e = poly(ctx, mul(ctx, z, valid), exp_coeffs, EXP_MIN_LEVEL)
        e = mul(ctx, e, valid * (1.0 / np.exp(exp_interval)))   # 패딩 exp(0)=1 제거 + 상한 나눗셈을 한 번에
        es.append(e)
        part = fold(ctx, e, lay.B, lay.Fg)
        denom = part if denom is None else ctx.engine.add(denom, part)  # feature 그룹끼리 덧셈
    denom = mul(ctx, denom, lay.block0_mask(count) * (1.0 / n_valid))   # 첫 block만 남기고 y0=1/n 곱하기
    z_iter = spread(ctx, denom, lay.B, lay.Fg)
    # 빈 슬롯(없는 노드/패딩)은 z=0인데, 0은 z <- z(2-z)의 불안정 고정점이라 CKKS 노이즈가 음수 쪽이면 매 반복
    # ~2배씩 자라다 발산한다. 그 쓰레기가 다음 bootstrap에서 ciphertext 전체 슬롯을 오염시킨 게 2026-10-09 wine
    # fold0 1e80 폭주의 원인. 빈 슬롯에 1(안정 고정점)을 넣어 두면 노이즈가 감쇠한다. 유효 슬롯 값은 그대로.
    z_iter = ctx.engine.add(z_iter, 1.0 - lay._mask(count, lay.Fg, lay.B))
    ws = [mul(ctx, e, 1.0 / n_valid) for e in es]
    for _ in range(recip_iterations):
        z_iter, *ws = lvs(ctx, [z_iter] + ws)   # 분모와 분자들을 같이 레벨 확보(bootstrap 2개씩 merge)
        two_minus_z = ctx.engine.subtract(2.0, z_iter)
        ws = [ctx.engine.multiply(w, two_minus_z, ctx.rlk) for w in ws]
        z_iter = ctx.engine.multiply(z_iter, two_minus_z, ctx.rlk)
    return ws


# ---- 합성 다항식 clamp (ROADMAP B) ----
SIGN_K = 12
_F_COEFFS = [0.0, 1.5, 0.0, -0.5]  # f(y) = (3y - y^3)/2


def he_sign(ctx, z, inv_radius, k: int = SIGN_K):
    y = mul(ctx, z, inv_radius)
    for _ in range(k):
        y = poly(ctx, y, _F_COEFFS, 5)
    return y


def he_sign_many(ctx, zs: list, inv_radius, k: int = SIGN_K) -> list:
    """서로 독립인 여러 입력의 sign을 나란히 계산 - 반복마다 레벨 확보를 같이 해서 bootstrap을 2개씩 merge."""
    ys = [mul(ctx, z, inv_radius) for z in zs]
    for _ in range(k):
        ys = lvs(ctx, [ctx.engine.intt(y) for y in ys], 5)
        ys = [ctx.engine.evaluate_polynomial(y, _F_COEFFS, ctx.rlk) for y in ys]
    return ys


def he_clamp(ctx, x, bound, inv_radius, k: int = SIGN_K):
    """clamp(x, -bound, bound) = (|x+B| - |x-B|)/2, |z| = z*sign(z). bound/inv_radius는 슬롯별 평문 벡터 가능
    (threshold와 alpha/leaf가 같은 ciphertext에 섞여 있어도 한 번에 처리).
    2026-10-09: 예전엔 min(x,B) 다음 max(.,-B)를 순서대로 해서 sign 두 번이 직렬이었다. 이 식은 두 sign이
    서로 독립이라 나란히 계산하고 bootstrap을 merge할 수 있다(정확한 sign이면 두 식은 같은 값)."""
    eng = ctx.engine
    x = lv(ctx, x)
    zp, zm = eng.add(x, bound), eng.subtract(x, bound)
    sp, sm = he_sign_many(ctx, [zp, zm], inv_radius, k)
    ap, am = mul(ctx, zp, sp), mul(ctx, zm, sm)
    return lv(ctx, mul(ctx, eng.subtract(ap, am), 0.5))


def clamp_plain(x: np.ndarray, bound, inv_radius, k: int = SIGN_K) -> np.ndarray:
    """he_clamp과 같은 수식의 평문 버전(에뮬레이션/검증용)."""
    def sign(z):
        y = z * inv_radius
        for _ in range(k):
            y = (3 * y - y ** 3) / 2
        return y
    zp, zm = x + bound, x - bound
    return (zp * sign(zp) - zm * sign(zm)) / 2
