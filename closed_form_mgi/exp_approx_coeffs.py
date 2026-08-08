"""exp(-beta * x) polynomial approximation coefficients (soft-MGI용).

sigmoid_approx_coeffs.py와 같은 Chebyshev interpolation 방식이지만 타겟 함수가
sigmoid 대신 exp(-beta*x)이고, 입력 구간도 다르다 (soft_mgi.py 참고).

**구간을 왜 [0,1]로 고정했나**: soft-MGI의 입력은 division-free MGI score
(fully_encrypted_mgi_stump.encrypted_mgi_from_counts, = |S|^2 - sum_c|S_c|^2)를
공개 이론적 상한 n_samples^2로 나눈 값이다. |S|^2 - sum|S_c|^2 <= |S|^2 <= n_samples^2
이고 각 항이 항상 0 이상이므로(class 분포가 아무리 섞여도 |S_c|<=|S|), 정규화된 값은
**항상** [0,1] 안에 있다 (실측으로도 iris/wine/breast_cancer 전부 0.076~0.61 범위 -
EXPERIMENT_LOG.md soft-MGI 세션 참고). MinMaxScaler처럼 실제 batch의 min/max를 그때그때
찾으려면 ciphertext 비교가 필요해서(=sign_bootstrap 계열) "sign_bootstrap 없는 경로"라는
soft-MGI의 목적과 충돌하므로, 데이터에 의존하지 않는 이 공개 상한을 정규화 기준으로 쓴다.
"""

from __future__ import annotations

import numpy as np


EXP_APPROX_DEGREE = 20
EXP_APPROX_INTERVAL = (0.0, 1.0)
# degree=15였을 때는 beta>=20부터 max_error가 급격히 커져서(beta=25: 3.3e-5, beta=30: 1.4e-4,
# beta=50: 3.2e-3) encrypted weight의 sum이 1에서 벗어나고 개별 weight가 음수/1 초과로
# 발산했다(2026-08-05 beta sweep 디버깅, run_soft_mgi_stump.py 실행 로그 참고). degree를
# 20으로 올리면 beta<=50까지 max_error<1e-4 유지(추가 비용은 evaluate_polynomial 차수
# 15->20, CKKS 곱셈 depth 소폭 증가만). beta>=100은 degree를 100까지 올려도 cheb2poly
# 변환 자체가 double precision에서 수치적으로 붕괴해서(계수가 1e+45 스케일로 발산) 이
# 방식(monomial basis로 변환 후 evaluate_polynomial)으로는 도달 불가 - desilofhe의
# evaluate_chebyshev_polynomial(Chebyshev basis 그대로 평가, cheb2poly 변환 없음)로
# 갈아타야 하는 별도 작업이 필요하다.


def chebyshev_approximation_exp(
    degree: int,
    beta: float,
    interval: tuple[float, float] = EXP_APPROX_INTERVAL,
) -> np.ndarray:
    """exp(-beta*x) (x in interval)의 Chebyshev interpolation polynomial 계수.

    sigmoid_approx_coeffs.chebyshev_approximation()과 동일한 절차(Chebyshev node 생성 ->
    Chebyshev basis 계수 -> 표준 다항식 basis로 변환)를 exp(-beta*x)에 적용한 버전.
    """
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


def exp_neg_beta_coeffs(beta: float, degree: int = EXP_APPROX_DEGREE) -> list[float]:
    """evaluate_polynomial에 바로 넣을 수 있는 list 형태로 exp(-beta*x) 계수 반환."""
    return chebyshev_approximation_exp(degree, beta).tolist()


def max_approximation_error(
    coeffs: list[float] | np.ndarray,
    beta: float,
    interval: tuple[float, float] = EXP_APPROX_INTERVAL,
) -> float:
    a, b = interval
    x_test = np.linspace(a, b, 20001)
    y_true = np.exp(-beta * x_test)
    y_approx = np.polynomial.polynomial.polyval(x_test, coeffs)
    return float(np.max(np.abs(y_true - y_approx)))


def main() -> None:
    """low/medium/high beta 각각에서 근사 오차를 확인 (run_soft_mgi_stump.py의 sweep과 맞춤)."""
    for beta in (1.0, 4.0, 10.0):
        coeffs = exp_neg_beta_coeffs(beta)
        err = max_approximation_error(coeffs, beta)
        print(f"beta={beta:5.1f} | degree={EXP_APPROX_DEGREE} | interval={EXP_APPROX_INTERVAL} | max error={err:.3e}")


if __name__ == "__main__":
    main()
