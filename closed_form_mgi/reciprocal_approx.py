"""Encrypted 1/D 근사 (Newton-Raphson).

**경고 (2026-08-05 확정): `engine.bootstrap()`은 ciphertext 값의 크기가 대략 2~5를
넘으면 에러 없이 조용히 값을 깨뜨린다** (독립 측정: 0.5->오차 0.02%, 10->15%, 22.8->65%,
50->120%(부호까지 뒤집힘) - bootstrap_magnitude_test.log 참고. `sign_bootstrap`이 입력을
[-1,1]로 정규화해야 하는 것과 같은 종류의 제약이다). 이 파일의 `y=1/D`는 D가 작을수록
(=이 함수를 큰 값에 쓸수록) 커지므로, iterations를 늘려서 level이 떨어져 bootstrap이
걸리는 순간 값이 깨질 위험이 항상 있다 - **작은 값(대략 2 이하)으로만 남는다고 보장할 수
있는 경우가 아니면 이 함수를 그대로 쓰지 말 것**. soft_mgi.py는 이 문제 때문에
encrypted_reciprocal을 더 이상 쓰지 않고, y 대신 z=D*y/w=exp_val*y(둘 다 (0,1)에
갇힘)를 추적하는 방식으로 바꿨다 - 새로 쓸 때도 같은 패턴(추적하는 양이 항상 bootstrap
안전범위 안에 있도록 재구성)을 우선 고려할 것.


desilofhe Engine에는 division/reciprocal 내장 연산이 없다(`evaluate_polynomial`/
`evaluate_chebyshev_polynomial`만 있음 - Engine 전체 메서드 목록으로 확인, 2026-08-05
soft-MGI 세션). client_assisted 프로토콜은 client가 secret key로 decrypt한 뒤 plaintext
numpy 나눗셈을 쓰기 때문에(client_ops.py의 weighted_gini_from_counts) 이 문제 자체가
없었다 - fully-encrypted 경로(soft-MGI 포함)에서 처음 필요해진 연산이다.

**왜 Chebyshev 다항식 fit이 아니라 Newton-Raphson인가**: soft_mgi.py의 분모
D = sum_i exp(-beta*x_i)는 beta가 커질수록(sharp한 softmax) 최소값이 지수적으로
작아진다(예: n_valid=16, beta=10 -> D_lo~=7.3e-4, D_hi=16 - 4자리 이상 차이나는 동적
범위). 이런 넓은 구간에서 1/x를 고정 차수 Chebyshev 다항식 하나로 fit하면 x->0 근처
근사오차가 발산해서 degree를 아무리 올려도 감당이 안 된다. Newton-Raphson(y_{n+1} =
y_n*(2 - D*y_n))은 초기값만 안전 범위(0 < y0 < 2/D) 안에 있으면 반복마다 오차가 제곱으로
줄어들어(quadratic convergence) 동적 범위 문제에 영향을 안 받는다 - 그래서 안전한 고정
초기값 하나로 low/medium/high beta 전부를 커버할 수 있다.
"""

from __future__ import annotations

from closed_form_mgi.primitives import ensure_level


def safe_initial_guess(d_upper_bound: float) -> float:
    """y0 = 1/d_upper_bound.

    D는 항상 [0, d_upper_bound] 안에 있으므로(soft_mgi.py 참고, d_upper_bound=n_valid),
    y0 = 1/d_upper_bound는 항상 0 < y0 <= 1/D(=<2/D)를 만족해서 Newton-Raphson이 발산하지
    않는 안전한 시작점이다 (D가 공개 이론적 상한 안에만 있으면 되므로 y0 자체는 데이터를
    보지 않고 정할 수 있는 public 상수).
    """
    return 1.0 / d_upper_bound


def encrypted_reciprocal(ctx, enc_d, d_upper_bound: float, iterations: int = 35, min_level: int = 8):
    """enc_d(스칼라, 모든 슬롯에 broadcast돼 있다고 가정 - engine.sum() 출력과 호환)의
    encrypted 1/x를 Newton-Raphson으로 근사.

    **iterations를 "여유있게 크게" 잡으면 오히려 더 나빠진다 (2026-08-05 beta sweep
    디버깅에서 발견)**: 처음엔 이론상 최악의 경우(D가 극단적으로 작은 경우)를 가정해서
    iterations=80(넉넉한 여유)로 뒀었는데, beta=20/30/50에서 encrypted 결과가
    (exp 다항식 자체는 decrypt해서 확인해보니 평문과 1e-7 이내로 정확했음에도 불구하고)
    완전히 틀리거나(beta=20: 1/D가 22.79가 아니라 20.06) 폭주했다(beta=50: weight가
    수만 단위로 발산). 원인은 평문 Newton-Raphson은 한 번 수렴하면(y=1/D) 그 이후
    반복해도 고정점이라 전혀 안 움직이지만, encrypted 버전은 매 반복이 실제
    ciphertext-ciphertext multiply 2번 + 필요시 bootstrap(근사 연산, noise가
    누적됨)을 수행한다 - 그래서 "이미 수렴한 뒤에 도는 반복"은 수학적으로는 공짜지만
    암호화 상태에서는 계속 noise만 쌓는 순비용이다. 실제 사용되는 D는(정규화된 MGI
    score가 이론적 극단([0,1])이 아니라 실측으로 23~56%에 몰려있으므로) worst-case로
    가정했던 것보다 훨씬 커서 15~30회 안에 이미 수렴한다(beta=20:15회, 30:20회,
    50:25회, 60:30회 - 전부 plaintext로 정밀 확인). iterations=35는 이 실측 수렴
    속도에 여유를 조금만 더한 값이다 - 필요 이상으로 키우지 말 것(noise만 쌓인다).
    """
    y = ctx.engine.encrypt([safe_initial_guess(d_upper_bound)] * ctx.engine.slot_count, ctx.pk)
    d = enc_d
    for _ in range(iterations):
        d = ensure_level(ctx, d, min_level=min_level)
        y = ensure_level(ctx, y, min_level=min_level)
        dy = ctx.engine.multiply(d, y, ctx.rlk)
        two_minus_dy = ctx.engine.subtract(2.0, dy)
        y = ctx.engine.multiply(y, two_minus_dy, ctx.rlk)
    return y
