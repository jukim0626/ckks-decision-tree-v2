"""Encrypted sigmoid 다항식 근사 (soft decision tree의 gate에 쓰는 핵심 비선형).
CKKS는 진짜 sigmoid(비교 연산)를 지원하지 않으므로 Chebyshev 다항식으로 근사한다.

**주의**: 다항식은 fit interval 밖에서 값이 발산한다(수학적으로 어떤 유한 구간 밖에서도
결국 발산 - 완전히 안전한 다항식은 원천적으로 불가능). interval을 벗어나는 입력이 나올
위험이 있는 실험(예: 여러 feature를 그대로 더하는 오블리크 gate)에서는 gate 입력이
안전 범위 안에 있는지 별도로 확인/제어해야 한다 - archive/oblique의 발산 디버깅 기록 참고.

2026-09-09 리팩터로 ckks_tree.py + sigmoid_approx_coeffs.py에서 분리됨."""

from __future__ import annotations

import numpy as np
from desilofhe import Engine

STEEPNESS = 8.0  # sigmoid(steepness*x)의 steepness - SPLIT_SIGMOID_COEFFS에 이미 baked-in.
# steepness=1(원래 sigmoid)은 threshold 근처 변화가 너무 완만해서 split 선택이 흐려짐 -
# steepness=8로 hard-step에 가깝게 만들면 정확도가 크게 개선된다(EXPERIMENT_LOG.md
# 2026-07-08 "sigmoid steepness" 항목 참고).

SIGMOID_APPROX_DEGREE = 15
SIGMOID_APPROX_INTERVAL = 2.0

# 15차 Chebyshev interpolation 근사 계수, interval=[-2, 2], steepness=8.
# sigmoid(steepness * x) ~= sum(SPLIT_SIGMOID_COEFFS[i] * x**i)
SPLIT_SIGMOID_COEFFS = [
    0.5,
    1.8573106551235292,
    0.0,
    -5.175600615181846,
    0.0,
    8.696566856169511,
    0.0,
    -7.9303964373277935,
    0.0,
    4.069569546220039,
    0.0,
    -1.175453917197851,
    0.0,
    0.1782425930260525,
    0.0,
    -0.011031284599643119,
]


def true_sigmoid(x: np.ndarray) -> np.ndarray:
    """실제 sigmoid 값 (근사 검증/오차 분석용)."""
    return 1.0 / (1.0 + np.exp(-x))


def chebyshev_approximation(degree: int, interval: float, steepness: float = 1.0) -> np.ndarray:
    """sigmoid(steepness * x) (x in [-interval, interval])의 Chebyshev interpolation
    polynomial 계수."""
    a, b = -interval, interval
    n_nodes = degree + 1
    k = np.arange(n_nodes)

    cheb_nodes_std = np.cos((2 * k + 1) * np.pi / (2 * n_nodes))
    cheb_nodes = 0.5 * (a + b) + 0.5 * (b - a) * cheb_nodes_std
    func_values = true_sigmoid(steepness * cheb_nodes)

    cheb_coeffs = np.zeros(n_nodes)
    for j in range(n_nodes):
        basis_values = np.cos(j * np.arccos(cheb_nodes_std))
        cheb_coeffs[j] = (2.0 / n_nodes) * np.sum(func_values * basis_values)
    cheb_coeffs[0] /= 2.0

    poly_coeffs_std = np.polynomial.chebyshev.cheb2poly(cheb_coeffs)
    scale = 2.0 / (b - a)
    shift = -(a + b) / (b - a)
    transformed = np.polynomial.polynomial.Polynomial(poly_coeffs_std)
    final_poly = transformed(np.polynomial.polynomial.Polynomial([shift, scale]))
    return final_poly.coef


def max_approximation_error(
    coeffs: list[float] | np.ndarray,
    interval: float,
    steepness: float = 1.0,
) -> float:
    """지정 구간에서 sigmoid(steepness * x) polynomial의 최대 절대 오차를 계산(진단용)."""
    x_test = np.linspace(-interval, interval, 20001)
    y_true = true_sigmoid(steepness * x_test)
    y_approx = np.polynomial.polynomial.polyval(x_test, coeffs)
    return float(np.max(np.abs(y_true - y_approx)))


def sigmoid_approx_enc(engine: Engine, rlk, enc_x, coeffs: list[float] | None = None):
    """설정된 다항식 계수로 sigmoid를 근사. evaluate_polynomial은 coeffs[i]를 i차 계수로 받는다."""
    if coeffs is None:
        coeffs = SPLIT_SIGMOID_COEFFS
    return engine.evaluate_polynomial(enc_x, coeffs, rlk)
