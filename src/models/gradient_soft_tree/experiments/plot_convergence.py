"""데이터셋별 iteration-정확도 수렴 그래프(2026-10-17, 논문 그림 후보).

입력:
  results/iteration_budget/curves.jsonl  - iteration_budget_study.py의 CKKS 다항식 에뮬레이션(10 iteration 간격, 300까지)
  results/paper_gpu/final_runs.jsonl     - GPU CKKS 본 실험의 중간 평가(30/50/100, 다중 클래스는 100/200/300)
수렴 지점 t_conv: fold 평균 test acc(앙상블)가 plateau(t=260~300 평균) - TOL에 처음 도달한 iteration.

python -m models.gradient_soft_tree.experiments.plot_convergence <out_dir>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
CURVES = ROOT / "results/iteration_budget/curves.jsonl"
GPU = ROOT / "results/paper_gpu/final_runs.jsonl"
TOL = 0.02  # 1%p는 wine(test 59개, 1샘플=1.7%p) 진동을 못 견딤
NAMES = {"breast_cancer_nominal": "breast_cancer(nominal)", "soybean15": "soybean", "cover10k": "covertype(10k)"}


def load_curves():
    acc = defaultdict(lambda: defaultdict(dict))  # ds -> t -> fold -> acc
    for line in open(CURVES):
        r = json.loads(line)
        acc[r["dataset"]][r["t"]][r["fold"]] = r["acc_ens"]
    out = {}
    for ds, by_t in acc.items():
        ts = sorted(by_t)
        a = np.array([[by_t[t][f] for f in sorted(by_t[t])] for t in ts])
        out[ds] = (np.array(ts), a.mean(1), a.std(1))
    return out


def load_gpu():
    acc = defaultdict(lambda: defaultdict(list))  # ds -> t -> [acc]
    if GPU.exists():
        for line in open(GPU):
            r = json.loads(line)
            f = r.get("finalize") or {}
            if "iterations" in f and "test_acc_poly" in f:
                acc[r["dataset"]][f["iterations"]].append(f["test_acc_poly"])
    return acc


def t_conv(ts, mean, tol=TOL):
    """plateau(마지막 5개 점 평균) - tol에 처음 도달한 iteration. "끝까지 유지"로 정의하면 wine처럼 t=270의
    한 번 딥(test 1~2샘플)만으로 수렴 지점이 280으로 밀려서, 처음 도달 기준을 쓴다."""
    plateau = mean[-5:].mean()
    return int(ts[np.argmax(mean >= plateau - tol)])


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    curves, gpu = load_curves(), load_gpu()
    order = [d for d in ["iris", "wine", "breast_cancer", "breast_cancer_nominal", "diabetes", "banknote",
                         "sepsis", "soybean15", "digits", "cover10k"] if d in curves]
    rows = []
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5), sharex=True)
    for ax, ds in zip(axes.flat, order):
        ts, m, s = curves[ds]
        tc = t_conv(ts, m)
        rows.append({"dataset": ds, "t_conv": tc, "acc_at_conv": float(m[ts == tc][0]), "acc_300": float(m[-1]), "plateau": float(m[-5:].mean()),
                     "acc_100": float(m[ts == 100][0]), "acc_30": float(m[ts == 30][0])})
        ax.plot(ts, m, color="C0", label="emulation (fold mean)")
        ax.fill_between(ts, m - s, m + s, color="C0", alpha=0.15)
        ax.axvline(tc, color="C3", ls="--", lw=1)
        ax.text(tc, ax.get_ylim()[0], f" {tc}", color="C3", va="bottom", fontsize=9)
        if gpu.get(ds):
            g = sorted(gpu[ds].items())
            ax.plot([t for t, _ in g], [np.mean(v) for _, v in g], "o", color="C1", ms=5,
                    label="CKKS GPU (fold mean)")
        ax.set_title(NAMES.get(ds, ds), fontsize=11)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("iteration")
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy")
    for ax in axes.flat[len(order):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Test accuracy vs. iteration (dashed: first iteration within {TOL*100:.0f}%p of plateau)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=200)
    fig.savefig(out / "convergence.pdf")
    with open(out / "convergence.json", "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
