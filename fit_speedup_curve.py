"""candidate 수(n) 대비 client-assisted/fully-encrypted MGI 시간에 curve fitting해서
배율(ratio)이 candidate 수가 늘어나면 어떻게 되는지 추세를 정량화.

모델:
  client-assisted: O(n) 순차 후보 평가 -> ca_time(n) = a*n + c
  fully-encrypted:  O(n) candidate scoring + O(log2 n) SIMD reduction round
                     -> mgi_time(n) = b*n + d*log2(n) + e
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

# bench_all_datasets.py 실측값 (iris/wine/breast_cancer/diabetes, sharpen=12, merge_bootstrap 적용)
DATA = {
    "iris": {"n": 12, "ca_time": 9.38, "mgi_time": 191.72},
    "wine": {"n": 39, "ca_time": 50.10, "mgi_time": 352.59},
    "breast_cancer": {"n": 90, "ca_time": 49.67, "mgi_time": 394.84},
    "diabetes": {"n": 24, "ca_time": 13.50, "mgi_time": 182.28},
    "digits": {"n": 192, "ca_time": 390.57, "mgi_time": 1237.02},
}


def ca_model(n, a, c):
    return a * n + c


def mgi_model(n, b, d, e):
    return b * n + d * np.log2(n) + e


def fit_and_report():
    names = list(DATA.keys())
    ns = np.array([DATA[k]["n"] for k in names], dtype=float)
    ca_times = np.array([DATA[k]["ca_time"] for k in names], dtype=float)
    mgi_times = np.array([DATA[k]["mgi_time"] for k in names], dtype=float)

    ca_params, _ = curve_fit(ca_model, ns, ca_times, p0=[0.5, 5.0])
    mgi_params, _ = curve_fit(mgi_model, ns, mgi_times, p0=[1.0, 20.0, 50.0], maxfev=10000)

    print("=== fitted 모델 ===")
    print(f"ca_time(n)  = {ca_params[0]:.4f}*n + {ca_params[1]:.4f}")
    print(f"mgi_time(n) = {mgi_params[0]:.4f}*n + {mgi_params[1]:.4f}*log2(n) + {mgi_params[2]:.4f}")
    print()
    print("=== fit 적합도 (실측 vs 예측) ===")
    for name, n, ca, mgi in zip(names, ns, ca_times, mgi_times):
        ca_pred = ca_model(n, *ca_params)
        mgi_pred = mgi_model(n, *mgi_params)
        print(
            f"{name:14s} n={n:5.0f} | ca 실측={ca:7.2f} 예측={ca_pred:7.2f} | "
            f"mgi 실측={mgi:7.2f} 예측={mgi_pred:7.2f} | ratio 실측={mgi/ca:.2f}x 예측={mgi_pred/ca_pred:.2f}x"
        )

    print("\n=== extrapolation: candidate 수를 늘리면 배율이 어떻게 되나 ===")
    for n in [12, 24, 39, 90, 150, 200, 300, 500, 1000, 2000, 5000, 10000]:
        ca_pred = ca_model(n, *ca_params)
        mgi_pred = mgi_model(n, *mgi_params)
        ratio = mgi_pred / ca_pred
        print(f"  n={n:6.0f} | ca_pred={ca_pred:9.2f}s | mgi_pred={mgi_pred:9.2f}s | ratio={ratio:6.2f}x")

    # 배율이 1배(breakeven)에 도달하는 n을 이분탐색으로 추정 (모델 유효범위 밖일 수 있음을 감안)
    lo, hi = 12.0, 1e9
    if mgi_model(hi, *mgi_params) / ca_model(hi, *ca_params) < 1.0:
        for _ in range(200):
            mid = (lo * hi) ** 0.5
            ratio = mgi_model(mid, *mgi_params) / ca_model(mid, *ca_params)
            if ratio > 1.0:
                lo = mid
            else:
                hi = mid
        print(f"\nbreakeven 추정 candidate 수: n ~= {hi:.0f} (모델 extrapolation, 외삽 신뢰도 낮음 주의)")
    else:
        print("\n이 모델 범위에서는 candidate 수를 아무리 늘려도 breakeven(1배)에 도달하지 않음")
        print("(mgi_time의 선형항 b*n이 ca_time의 선형항 a*n보다 커서, n이 커질수록 격차가 좁아지긴 해도 역전은 안 됨)")

    return ca_params, mgi_params, names, ns, ca_times, mgi_times


if __name__ == "__main__":
    fit_and_report()
