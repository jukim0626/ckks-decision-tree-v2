"""Sigmoid polynomial approximation coefficients.

DESILO FHE의 ``evaluate_polynomial``은 ``coeffs[i]``를 x^i 계수로 사용한다.
따라서 아래 리스트는 일반 polynomial basis 기준이다.
"""

import numpy as np


SIGMOID_APPROX_DEGREE = 5
SIGMOID_APPROX_INTERVAL = 8.0

# 5차 Chebyshev interpolation 근사 계수, interval=[-8, 8].
# sigmoid(x) ~= sum(SIGMOID_COEFFS[i] * x**i)
SIGMOID_COEFFS = [
    0.5,
    0.209637,
    0.0,
    -0.005402,
    0.0,
    0.000050,
]


def true_sigmoid(x: np.ndarray) -> np.ndarray:
    """실제 sigmoid 값을 계산."""
    return 1.0 / (1.0 + np.exp(-x))


def chebyshev_approximation(
    degree: int,
    interval: float,
) -> np.ndarray:
    """Chebyshev interpolation으로 sigmoid polynomial 계수를 계산."""
    a, b = -interval, interval
    n_nodes = degree + 1
    k = np.arange(n_nodes)

    cheb_nodes_std = np.cos((2 * k + 1) * np.pi / (2 * n_nodes))
    cheb_nodes = 0.5 * (a + b) + 0.5 * (b - a) * cheb_nodes_std
    func_values = true_sigmoid(cheb_nodes)

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
    coeffs = final_poly.coef

    coeffs[0] = 0.5
    coeffs[2::2] = 0.0
    return coeffs


def max_approximation_error(coeffs: list[float] | np.ndarray, interval: float) -> float:
    """지정 구간에서 sigmoid polynomial의 최대 절대 오차를 계산."""
    x_test = np.linspace(-interval, interval, 20001)
    y_true = true_sigmoid(x_test)
    y_approx = np.polynomial.polynomial.polyval(x_test, coeffs)
    return float(np.max(np.abs(y_true - y_approx)))


def main() -> None:
    """현재 설정의 coefficient와 근사 오차를 출력."""
    coeffs = chebyshev_approximation(
        degree=SIGMOID_APPROX_DEGREE,
        interval=SIGMOID_APPROX_INTERVAL,
    )

    print("=== Chebyshev sigmoid approximation ===")
    print(f"degree: {SIGMOID_APPROX_DEGREE}")
    print(f"interval: [-{SIGMOID_APPROX_INTERVAL:g}, {SIGMOID_APPROX_INTERVAL:g}]")
    print("coeffs:")
    for degree, coeff in enumerate(coeffs):
        print(f"  x^{degree:02d}: {coeff:.18e}")

    max_error = max_approximation_error(coeffs, SIGMOID_APPROX_INTERVAL)
    print(f"max error: {max_error:.12f}")


if __name__ == "__main__":
    main()
