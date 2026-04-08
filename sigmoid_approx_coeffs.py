import numpy as np


def true_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def chebyshev_approximation(func, degree, interval):
    """
    Chebyshev 근사로 다항식 계수 구하기

    func     : 근사할 함수 (sigmoid)
    degree   : 다항식 차수 (5)
    interval : 근사 구간 [-a, a]
    """
    a, b = -interval, interval

    # Chebyshev 노드 계산
    # [-1, 1] 구간의 Chebyshev 노드를 [a, b]로 변환
    # k번째 노드: cos((2k+1)π / 2(n+1))
    n = degree + 1  # 노드 개수
    k = np.arange(n)
    # [-1, 1] 구간 노드
    cheb_nodes_std = np.cos((2 * k + 1) * np.pi / (2 * n))
    # [a, b] 구간으로 변환
    cheb_nodes = 0.5 * (a + b) + 0.5 * (b - a) * cheb_nodes_std

    # 각 노드에서 함수값 계산
    func_values = func(cheb_nodes)

    # Chebyshev 계수 계산 (이산 코사인 변환 방식)
    # cj = (2/n) * Σ f(xk) * Tj(xk_std)
    cheb_coeffs = np.zeros(n)
    for j in range(n):
        Tj = np.cos(j * np.arccos(cheb_nodes_std))  # Chebyshev 다항식 값
        cheb_coeffs[j] = (2 / n) * np.sum(func_values * Tj)
    cheb_coeffs[0] /= 2  # c0는 절반

    # Chebyshev 계수 → 일반 다항식 계수로 변환
    # np.polynomial.chebyshev.cheb2poly 사용
    # 단, 구간 변환 때문에 표준화된 변수로 변환 필요
    poly_coeffs_std = np.polynomial.chebyshev.cheb2poly(cheb_coeffs)

    # [a, b] → [-1, 1] 변환을 반영한 계수 보정
    # x_std = (2x - (a+b)) / (b-a)
    # x = x_std * (b-a)/2 + (a+b)/2
    scale = 2 / (b - a)
    shift = -(a + b) / (b - a)

    # 변환된 계수를 원래 x에 대한 계수로 변환
    from numpy.polynomial.polynomial import polyfromroots, polyval
    transformed = np.polynomial.polynomial.Polynomial(poly_coeffs_std)
    # x_std = scale*x + shift 대입
    final_poly = transformed(np.polynomial.polynomial.Polynomial([shift, scale]))
    final_coeffs = final_poly.coef

    return final_coeffs, cheb_coeffs


# 5차 Chebyshev 근사, 구간 [-8, 8]
degree = 5
interval = 8
final_coeffs, cheb_coeffs = chebyshev_approximation(true_sigmoid, degree, interval)

print("=== Chebyshev 근사 결과 ===")
print(f"상수항 (x⁰): {final_coeffs[0]:.6f}")
print(f"x¹ 계수:     {final_coeffs[1]:.6f}")
print(f"x² 계수:     {final_coeffs[2]:.6f}")
print(f"x³ 계수:     {final_coeffs[3]:.6f}")
print(f"x⁴ 계수:     {final_coeffs[4]:.6f}")
print(f"x⁵ 계수:     {final_coeffs[5]:.6f}")

# 최대 오차 계산
x_test = np.linspace(-interval, interval, 10000)
y_true = true_sigmoid(x_test)
y_approx = sum(final_coeffs[i] * x_test ** i for i in range(len(final_coeffs)))
max_err = np.max(np.abs(y_true - y_approx))
print(f"\n최대 오차: {max_err:.6f}")

