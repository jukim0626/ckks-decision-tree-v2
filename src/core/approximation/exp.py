"""exp(-beta*x) Chebyshev 다항식 근사 계수 생성. sigmoid_approx_coeffs.py와 같은 방식으로
Chebyshev interpolation을 하되 타겟 함수가 exp(-beta*x). softmax(core/softmax.py)와
MGI 계보(archive/closed_form_mgi)가 공유하는 프리미티브.

2026-09-09 리팩터로 closed_form_mgi/exp_approx_coeffs.py에서 분리됨(generic 함수만)."""

from __future__ import annotations

import numpy as np


def chebyshev_approximation_exp(
    degree: int,
    beta: float,
    interval: tuple[float, float],
) -> np.ndarray:
    """exp(-beta*x) (x in interval)의 Chebyshev interpolation polynomial 계수.
    beta에 음수를 주면 exp(+|beta|*x)를 근사할 수 있다."""
    a, b = interval
    n_nodes = degree + 1
    k = np.arange(n_nodes)

    cheb_nodes_std = np.cos((2 * k + 1) * np.pi / (2 * n_nodes))
    cheb_nodes = 0.5 * (a + b) + 0.5 * (b - a) * cheb_nodes_std
    func_values = np.exp(-beta * cheb_nodes)

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


def chebyshev_approximation_exp_plain(degree: int, interval: tuple[float, float]) -> np.ndarray:
    """exp(x)(x in interval, beta 없는 단순 버전)의 Chebyshev interpolation polynomial 계수.

    2026-09-15: softmax.py가 매번 "beta=-1.0을 줘서 exp(+x)를 얻는다"는 부호 반전을
    설명해야 했던 게 헷갈린다는 피드백으로 신설 - beta 개념을 아예 감춘 얇은 wrapper.
    beta가 진짜로 필요한 archive/closed_form_mgi는 계속 chebyshev_approximation_exp를
    직접 쓰고, 이 함수는 손대지 않는다(같은 로직이라 결과 계수는 100% 동일)."""
    return chebyshev_approximation_exp(degree, beta=-1.0, interval=interval)
