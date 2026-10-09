"""global_opt_study 결과 분석.

python -m models.gradient_soft_tree.experiments.global_opt_analyze <out_dir>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from models.gradient_soft_tree.experiments.global_opt_study import DATASETS, _data  # noqa: E402

NEAR = 0.05   # 정규화 gap이 이 이하면 "최선값 근처"
BAD = 0.30    # 이 이상이면 "나쁜 run"


def main() -> None:
    rows = [json.loads(l) for l in open(Path(sys.argv[1]) / "runs.jsonl")]
    Lstar = {ds: min(r["L"] for r in rows if r["dataset"] == ds) for ds in DATASETS}
    for r in rows:
        r["gap"] = (r["L"] - Lstar[r["dataset"]]) / max(r["L0"] - Lstar[r["dataset"]], 1e-12)
    by = defaultdict(list)
    for r in rows:
        by[(r["method"]["name"], r["dataset"])].append(r)

    print("== 1. 현재 방법(base, seed 80개)의 최적성 gap - L*는 seed 50개 x 2000it 포함 전체 최저 ==")
    print(f"{'dataset':22s} {'L*':>7s} {'gap 평균':>8s} {'중앙':>6s} {'90%':>6s} {'근처(<=5%)':>10s} {'나쁜(>=30%)':>11s} "
          f"{'test acc':>13s} | {'long2000 gap':>12s} {'long acc':>8s}")
    for ds in DATASETS:
        b = by[("base", ds)]
        g = np.array([r["gap"] for r in b])
        acc = np.array([r["test_acc"] for r in b])
        lg = by[("long", ds)]
        print(f"{ds:22s} {Lstar[ds]:7.4f} {g.mean():8.3f} {np.median(g):6.3f} {np.quantile(g, .9):6.3f} "
              f"{(g <= NEAR).mean():10.0%} {(g >= BAD).mean():11.0%} {acc.mean():.3f}±{acc.std():.3f} | "
              f"{np.mean([r['gap'] for r in lg]):12.3f} {np.mean([r['test_acc'] for r in lg]):8.3f}")

    print("\n== 2. routing / gradient / 구간 (base) ==")
    for ds in DATASETS:
        b = by[("base", ds)]
        print(f"{ds:22s} 죽은 leaf {np.mean([r['routing']['dead_leaves'] for r in b]):.2f}  "
              f"leaf 엔트로피 {np.mean([r['routing']['leaf_entropy'] for r in b]):.2f}  "
              f"노드 불균형 {np.mean([r['routing']['node_imbalance'] for r in b]):.2f}  "
              f"|grad| thr {np.mean([r['grad_norm'].get('threshold', 0) for r in b]):.1e} "
              f"leaf {np.mean([r['grad_norm'].get('leaf_logits', 0) for r in b]):.1e}  "
              f"clamp비율 {np.mean([r['clamp_rate'] for r in b]):.2f}  "
              f"clamp전 최대|t| {np.max([r['pre_clamp_max']['threshold'] for r in b]):.2f}")
        # gap과 routing의 상관(나쁜 run이 routing 쏠림 때문인지)
    print("\n   gap vs 죽은 leaf 수 상관(spearman 근사): ", end="")
    for ds in DATASETS:
        b = by[("base", ds)]
        g = np.array([r["gap"] for r in b]); d = np.array([r["routing"]["dead_leaves"] for r in b])
        c = np.corrcoef(np.argsort(np.argsort(g)), np.argsort(np.argsort(d)))[0, 1] if d.std() > 0 else float("nan")
        print(f"{ds[:6]}={c:+.2f}", end=" ")
    print()

    print("\n== 3. multi-start R=8 (base seed 80개를 8개씩 10그룹) ==")
    print(f"{'dataset':22s} {'단일 acc':>9s} {'앙상블 acc':>11s} {'최선선택 acc':>12s} {'최선선택 gap':>12s} {'최선선택 근처':>12s}")
    for ds in DATASETS:
        _, _, _, yte, _, _ = _data(ds)
        b = sorted(by[("base", ds)], key=lambda r: r["seed"])
        ens, best_acc, best_gap = [], [], []
        for k in range(len(b) // 8):
            grp = b[8 * k: 8 * k + 8]
            yh = np.mean([np.array(r["yhat_te"]) for r in grp], axis=0)
            ens.append((yh.argmax(1) == yte).mean())
            best = min(grp, key=lambda r: r["L"])
            best_acc.append(best["test_acc"]); best_gap.append(best["gap"])
        print(f"{ds:22s} {np.mean([r['test_acc'] for r in b]):9.3f} {np.mean(ens):11.3f} {np.mean(best_acc):12.3f} "
              f"{np.mean(best_gap):12.3f} {np.mean(np.array(best_gap) <= NEAR):12.0%}")

    print("\n== 4. 방법별 비교 (seed 0~19, base와 같은 seed 짝비교, 데이터셋 평균) ==")
    names = sorted({r["method"]["name"] for r in rows} - {"long"}, key=lambda s: (s[0] != "b", s))
    print(f"{'method':18s} {'gap 평균':>8s} {'근처':>6s} {'나쁜':>6s} {'test acc':>9s} {'Δacc':>7s} {'gap 개선 데이터셋(p<.05)':>24s}")
    for nm in names:
        gaps, near, bad, acc, dacc, better = [], [], [], [], [], 0
        for ds in DATASETS:
            m = {r["seed"]: r for r in by[(nm, ds)] if r["seed"] < 20}
            b = {r["seed"]: r for r in by[("base", ds)] if r["seed"] < 20}
            seeds = sorted(set(m) & set(b))
            gm = np.array([m[s]["gap"] for s in seeds]); gb = np.array([b[s]["gap"] for s in seeds])
            gaps.append(gm.mean()); near.append((gm <= NEAR).mean()); bad.append((gm >= BAD).mean())
            am = np.array([m[s]["test_acc"] for s in seeds]); ab = np.array([b[s]["test_acc"] for s in seeds])
            acc.append(am.mean()); dacc.append((am - ab).mean())
            if nm != "base" and np.any(gm != gb):
                try:
                    if wilcoxon(gm, gb, alternative="less").pvalue < 0.05:
                        better += 1
                except ValueError:
                    pass
        print(f"{nm:18s} {np.mean(gaps):8.3f} {np.mean(near):6.0%} {np.mean(bad):6.0%} {np.mean(acc):9.3f} "
              f"{np.mean(dacc):+7.3f} {('-' if nm == 'base' else f'{better}/{len(DATASETS)}'):>24s}")


if __name__ == "__main__":
    main()
