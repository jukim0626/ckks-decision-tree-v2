"""Sigmoid polynomial approximation coefficients.

DESILO FHE의 ``evaluate_polynomial``은 ``coeffs[i]``를 x^i 계수로 사용한다.
따라서 아래 리스트는 일반 polynomial basis 기준이다.
"""

import numpy as np


SIGMOID_APPROX_DEGREE = 15
SIGMOID_APPROX_INTERVAL = 2.0
SIGMOID_APPROX_STEEPNESS = 8.0

# 15차 Chebyshev interpolation 근사 계수, interval=[-2, 2], steepness=8
# (sigmoid(steepness * x)를 근사 - 아래 "steepness가 왜 필요한가" 참고).
# sigmoid(steepness * x) ~= sum(SPLIT_SIGMOID_COEFFS[i] * x**i)
#
# steepness가 왜 필요한가: steepness=1(원래 sigmoid, x 그대로)은 threshold 근처에서
# 변화가 너무 완만해서, threshold와 좀만 떨어져도 0/1이 아니라 애매한 값(0.3~0.7)으로
# 배정된다. 이게 Gini 기반 split 선택 자체를 흐리게 만들어서, 실제로 제일 판별력 있는
# feature가 아니라 이 "흐릿함"과 우연히 잘 맞는 feature를 고르게 만든다 (실측: wine에서
# sklearn은 color_intensity/flavanoids를 쓰는데 steepness=1인 우리 tree는 그 두 feature를
# candidate grid를 1000개로 늘려도 단 한 번도 안 고르고 proline/od280만 반복 선택함).
# steepness=8로 sigmoid를 더 가파르게(hard-step에 더 가깝게) 만들면 세 데이터셋(iris/wine/
# breast_cancer) 전부 depth 3~5에서 평균 정확도가 56~70%대에서 90~100%대로 뛰어오른다
# (sklearn 표준 tree와 비슷한 수준). degree=15는 그대로라 CKKS 연산 비용 증가 없음.
# 상세 실험은 EXPERIMENT_LOG.md 2026-07-08 "sigmoid steepness" 항목 참고.
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

# 비교/argmax처럼 입력 범위가 더 넓은 곳을 위한 15차 근사, interval=[-8, 8], steepness=1.
# ckks_tree.py에 주석 처리된 encrypted_argmax()가 다시 활성화될 때를 대비해 남겨둔다.
COMPARE_SIGMOID_COEFFS = [
    0.5,
    0.2491195997302511,
    0.0,
    -0.01898333003953335,
    0.0,
    0.001318127024043555,
    0.0,
    -6.106184926773418e-05,
    0.0,
    1.7507805703225976e-06,
    0.0,
    -2.958878145465192e-08,
    0.0,
    2.6895813256413547e-10,
    0.0,
    -1.0116015493367081e-12,
]


def true_sigmoid(x: np.ndarray) -> np.ndarray:
    """실제 sigmoid 값을 계산."""
    return 1.0 / (1.0 + np.exp(-x))


def chebyshev_approximation(
    degree: int,
    interval: float,
    steepness: float = 1.0,
) -> np.ndarray:
    """Chebyshev interpolation으로 sigmoid(steepness * x) polynomial 계수를 계산."""
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
    coeffs = final_poly.coef

    coeffs[0] = 0.5
    coeffs[2::2] = 0.0
    return coeffs


def max_approximation_error(
    coeffs: list[float] | np.ndarray,
    interval: float,
    steepness: float = 1.0,
) -> float:
    """지정 구간에서 sigmoid(steepness * x) polynomial의 최대 절대 오차를 계산."""
    x_test = np.linspace(-interval, interval, 20001)
    y_true = true_sigmoid(steepness * x_test)
    y_approx = np.polynomial.polynomial.polyval(x_test, coeffs)
    return float(np.max(np.abs(y_true - y_approx)))


def main() -> None:
    """현재 설정의 coefficient와 근사 오차를 출력."""
    coeffs = chebyshev_approximation(
        degree=SIGMOID_APPROX_DEGREE,
        interval=SIGMOID_APPROX_INTERVAL,
        steepness=SIGMOID_APPROX_STEEPNESS,
    )

    print("=== Chebyshev sigmoid approximation ===")
    print(f"degree: {SIGMOID_APPROX_DEGREE}")
    print(f"interval: [-{SIGMOID_APPROX_INTERVAL:g}, {SIGMOID_APPROX_INTERVAL:g}]")
    print(f"steepness: {SIGMOID_APPROX_STEEPNESS:g}")
    print("coeffs:")
    for degree, coeff in enumerate(coeffs):
        print(f"  x^{degree:02d}: {coeff:.18e}")

    max_error = max_approximation_error(
        coeffs, SIGMOID_APPROX_INTERVAL, SIGMOID_APPROX_STEEPNESS
    )
    print(f"max error: {max_error:.12f}")


if __name__ == "__main__":
    main()
