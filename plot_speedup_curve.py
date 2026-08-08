"""candidate 수 대비 client-assisted vs fully-encrypted MGI 시간/배율을 실측점+fitted
curve+extrapolation으로 그린 그래프 (슬라이드용 PNG 저장)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from fit_speedup_curve import DATA, ca_model, mgi_model, fit_and_report


def main() -> None:
    ca_params, mgi_params, names, ns, ca_times, mgi_times = fit_and_report()

    # ca_model이 0 밑으로 내려가는(비현실적인) 구간은 fit 신뢰 범위 밖이므로 그래프에서 제외
    n_grid = np.linspace(max(min(ns) * 0.8, 1.0), 500, 400)
    ca_curve = ca_model(n_grid, *ca_params)
    valid = ca_curve > 1.0
    n_grid, ca_curve = n_grid[valid], ca_curve[valid]
    mgi_curve = mgi_model(n_grid, *mgi_params)
    ratio_curve = mgi_curve / ca_curve
    ratio_points = mgi_times / ca_times

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.scatter(ns, ca_times, color="#2b6cb0", label="client-assisted (실측)", zorder=5)
    ax1.scatter(ns, mgi_times, color="#c53030", label="fully-encrypted MGI (실측)", zorder=5)
    ax1.plot(n_grid, ca_curve, color="#2b6cb0", linestyle="--", alpha=0.6, label="client-assisted (fit)")
    ax1.plot(n_grid, mgi_curve, color="#c53030", linestyle="--", alpha=0.6, label="fully-encrypted (fit)")
    for name, n, ca, mgi in zip(names, ns, ca_times, mgi_times):
        ax1.annotate(name, (n, mgi), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax1.set_xlabel("candidate 수 (n)")
    ax1.set_ylabel("시간 (s)")
    ax1.set_title("학습 시간 vs candidate 수")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.scatter(ns, ratio_points, color="#2f855a", zorder=5, label="실측 배율")
    ax2.plot(n_grid, ratio_curve, color="#2f855a", linestyle="--", alpha=0.7, label="fitted 배율")
    ax2.axhline(1.0, color="gray", linestyle=":", label="배율=1 (breakeven)")
    for name, n, r in zip(names, ns, ratio_points):
        ax2.annotate(name, (n, r), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax2.set_xlabel("candidate 수 (n)")
    ax2.set_ylabel("배율 (fully-encrypted / client-assisted)")
    ax2.set_title("candidate 수가 늘수록 배율 격차가 좁혀지는 추세")
    ax2.set_ylim(0, max(ratio_points.max(), 5) * 1.3)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.text(
        0.02, 0.02, "n=5개 점 기반 fit - 외삽 신뢰도 낮음, 경향성 참고용",
        transform=ax2.transAxes, fontsize=7, color="gray", va="bottom",
    )

    fig.suptitle("client-assisted vs fully-encrypted MGI: candidate 수에 따른 속도 격차 추세", fontsize=12)
    fig.tight_layout()
    out_path = "/home/juhyun/projects/ckks-decision-tree-v2/speedup_vs_candidates.png"
    fig.savefig(out_path, dpi=150)
    print(f"\n그래프 저장: {out_path}")


if __name__ == "__main__":
    main()
