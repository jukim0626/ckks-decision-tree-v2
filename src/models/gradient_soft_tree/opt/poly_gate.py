"""Phase 4: degree-15 sigmoid approximation 대신 쓰는 low-degree native polynomial gate.

기존 SPLIT_SIGMOID_COEFFS(sigmoid_approx_coeffs.py)와 완전히 같은 방법(Chebyshev
interpolation, interval=[-2,2], steepness=8)으로 degree=5/3 계수를 새로 피팅한다 - "기존
sigmoid를 다항식으로 근사"가 아니라 "P(x-theta) 자체가 모델"이라는 문서의 conceptual
change에 맞춰, 이 계수로 만든 다항식 자체가 gate 함수가 된다 (baseline과 같은 steepness=8로
피팅해서 결정 경계 형태를 최대한 맞췄다 - 다른 steepness를 쓰면 gate sharpness 자체가
달라져서 degree 효과와 steepness 효과가 섞여버림).

required_level: opt/level_probe_gate.py로 실측한 "poly 진입 직전 level -> poly 직후 level"
소모량. degree=15 baseline(depthN_ckks.py 실측, 상단 주석 "poly는 레벨을 정확히 5
소모한다")과 같은 방법으로 degree=5/3도 실측해서 채웠다 (2026-08-27, level_preset=17,
CPU-side plaintext 근사 없이 실제 GPU evaluate_polynomial 호출로 측정).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from core.approximation.sigmoid import chebyshev_approximation, max_approximation_error

GATE_INTERVAL = 2.0
GATE_STEEPNESS = 8.0

# opt/level_probe_gate.py 실측 결과 (2026-08-27, level_preset=17, 실제 GPU
# evaluate_polynomial 호출): degree=15 -> 5 (depthN_ckks.py 주석의 기존 실측값과 정확히
# 일치, 프로브 자체의 신뢰성 확인). degree=5 -> 4, degree=3 -> 3.
REQUIRED_LEVEL_CONSUMED: dict[int, int] = {
    15: 5,
    5: 4,
    3: 3,
}

# depthN_ckks.py의 baseline guard(12)는 degree=15/consumed=5 기준으로
# "poly 진입 min_level - consumed >= _LOCAL_MIN_LEVEL(5) + 2(안전 버퍼)"를 만족하도록
# 고정된 값이었다(12-5=7=5+2, 상단 주석 "poly 직후 gate_j가 레벨 0으로 나올 수 있어서"
# 참고). degree를 낮췄는데도 이 12를 그대로 재사용하면 "P3인데도 level 12를 요구"하는
# 최적화 무의미 상태가 되므로, 같은 마진 공식을 degree마다 다시 적용해 guard를 낮춘다.
_LOCAL_MIN_LEVEL = 5
_SAFETY_BUFFER = 2


def gate_entry_min_level(degree: int) -> int:
    return REQUIRED_LEVEL_CONSUMED[degree] + _LOCAL_MIN_LEVEL + _SAFETY_BUFFER


@lru_cache(maxsize=None)
def gate_coeffs(degree: int) -> list[float]:
    coeffs = chebyshev_approximation(degree=degree, interval=GATE_INTERVAL, steepness=GATE_STEEPNESS)
    return coeffs.tolist()


@lru_cache(maxsize=None)
def gate_derivative_coeffs(degree: int) -> list[float]:
    """P(z)의 정확한 도함수 P'(z) 계수 (d/dz sum c_i z^i = sum i*c_i z^(i-1))."""
    coeffs = gate_coeffs(degree)
    deriv = [i * c for i, c in enumerate(coeffs)][1:]
    if not deriv:
        deriv = [0.0]
    return deriv


def gate_approx_error(degree: int) -> float:
    return max_approximation_error(gate_coeffs(degree), GATE_INTERVAL, GATE_STEEPNESS)


def required_level(degree: int) -> int:
    if degree not in REQUIRED_LEVEL_CONSUMED:
        raise KeyError(
            f"degree={degree}의 실측 level 소모량이 없다 - "
            f"opt/level_probe_gate.py --degree {degree} 먼저 실행할 것"
        )
    return REQUIRED_LEVEL_CONSUMED[degree]


def theoretical_mult_depth(degree: int) -> int:
    """Horner 평가 기준 이론적 곱셈 깊이 = degree (참고용, evaluate_polynomial의 실제 내부
    구현(power-tree 등)에 따라 실측 level 소모(required_level)와 다를 수 있음 - 실측을
    우선한다)."""
    return degree


if __name__ == "__main__":
    for d in (15, 5, 3):
        c = gate_coeffs(d)
        err = gate_approx_error(d)
        print(f"degree={d:2d}  max_abs_error_vs_sigmoid(steepness=8,[-2,2])={err:.6f}  coeffs={np.round(c, 6)}")
