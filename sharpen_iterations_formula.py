"""sign_bootstrap(x) ~= sin(pi/2 * x)를 k번 합성했을 때, 정규화된 score gap이 얼마나
sharp하게(=target_confidence 이상) saturate하는지에 필요한 최소 반복 횟수를 닫힌 형태로
추정하는 공식.

배경: fully_encrypted_mgi_simd_argmin.py의 sharpen_iterations를 8/12/20으로 실험을 통해
찾았는데(EXPERIMENT_LOG.md 2026-07-28), 이걸 "0 근처에서 sin(pi/2*x) 합성이 (pi/2)^k * x로
근사된다"는 성질에서 닫힌 형태로 유도한다 (0에서의 미분 pi/2 ~= 1.5708 > 1이라 반복 합성마다
값이 등비수열로 커짐 - 실제 sin이 포화되기 전까지는).
"""

from __future__ import annotations

import numpy as np

# target_confidence별로 "선형근사 (pi/2)^k*x가 tau를 넘으면 실제 값이 target_confidence를
# 넘는다"는 tau를 보수적으로(=여러 x0에서 실측한 값 중 최댓값) calibration한 테이블.
# calibrate_tau_table()로 재생성 가능 (아래 참고).
TAU_TABLE = {0.50: 1.0, 0.90: 2.0637, 0.95: 2.3721, 0.99: 3.3897}


def _iterate(x0: float, k: int) -> float:
    x = x0
    for _ in range(k):
        x = np.sin(np.pi / 2 * x)
    return x


def calibrate_tau_table(
    targets: tuple[float, ...] = (0.50, 0.90, 0.95, 0.99),
    x0_grid: tuple[float, ...] = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.064, 0.1),
) -> dict[float, float]:
    """여러 x0에서 target_confidence에 도달하는 데 필요한 실제 k를 찾고, 그 시점의
    선형근사값 (pi/2)^k*x0 중 최댓값(보수적)을 tau로 잡는다."""
    table = {}
    for target in targets:
        taus = []
        for x0 in x0_grid:
            k, x = 0, x0
            while x < target and k < 200:
                x = np.sin(np.pi / 2 * x)
                k += 1
            taus.append((np.pi / 2) ** k * x0)
        table[target] = max(taus)
    return table


def _tau_for_confidence(target: float) -> float:
    xs = sorted(TAU_TABLE.keys())
    if target <= xs[0]:
        return TAU_TABLE[xs[0]]
    if target >= xs[-1]:
        return TAU_TABLE[xs[-1]]
    for a, b in zip(xs, xs[1:]):
        if a <= target <= b:
            ta, tb = TAU_TABLE[a], TAU_TABLE[b]
            frac = (target - a) / (b - a)
            return ta + frac * (tb - ta)
    raise AssertionError


def required_sharpen_iterations(min_gap: float, target_confidence: float = 0.95) -> int:
    """정규화된 score gap(min_gap, (0,1) 근처의 작은 양수)이 sign_bootstrap을 k번 반복
    합성했을 때 target_confidence 이상으로 saturate하는 데 필요한 최소 k.

    k = ceil( log(tau / min_gap) / log(pi/2) )
    """
    if min_gap <= 0:
        raise ValueError("min_gap must be positive")
    tau = _tau_for_confidence(target_confidence)
    k = np.log(tau / min_gap) / np.log(np.pi / 2)
    return max(1, int(np.ceil(k)))


def actual_confidence_at(min_gap: float, k: int) -> float:
    """검증용: min_gap을 k번 sin(pi/2*x) 합성했을 때 실제 도달하는 값."""
    return _iterate(min_gap, k)


if __name__ == "__main__":
    print("=== 공식 검증: iris real_gap(hard-split)=0.0644 ===")
    for target in [0.90, 0.95]:
        k = required_sharpen_iterations(0.0644, target)
        print(f"  target={target}: k={k} (실제 도달값={actual_confidence_at(0.0644, k):.4f})")

    print("=== 공식 검증: iris tie-noise gap(hard-split)=0.0005 ===")
    for target in [0.90, 0.95]:
        k = required_sharpen_iterations(0.0005, target)
        print(f"  target={target}: k={k} (실제 도달값={actual_confidence_at(0.0005, k):.4f})")
    print(f"  참고: k=12일때={actual_confidence_at(0.0005, 12):.4f}, k=20일때={actual_confidence_at(0.0005, 20):.4f}")
