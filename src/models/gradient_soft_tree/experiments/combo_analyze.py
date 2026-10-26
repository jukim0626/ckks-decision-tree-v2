"""combo_study 결과 분석(2026-10-27). 조합 선택은 전부 inner-val로만 하고 test로 보고한다.

선택 방식:
  baseline   본실험 설정 mse_lam1_minmax
  global     모든 데이터셋에 같은 조합 1개 - 데이터셋별 val 점수(fold·seed 평균) 평균이 최고인 조합
  per_ds     (dataset, fold, seed)마다 val 점수 최고 조합(동점이면 baseline에 가까운 쪽 = CONFIG 순서상 앞)
  rule       소수 클래스 비율 < 10%면 inv, 아니면 mse (λ/스케일링은 global 조합 것)
  oracle     test 점수 최고 조합(참고용 상한 - 보고 금지)
선택 지표는 MCC(기본)와 accuracy 두 가지로 따로 돌려 민감도를 본다. iteration은 예산(K<=3: 100, 다중: 300)과 500.

python -m models.gradient_soft_tree.experiments.combo_analyze ../results/combo_study
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from models.gradient_soft_tree.experiments.ablation_study import DATASETS
from models.gradient_soft_tree.experiments.combo_study import CONFIGS

BASE = "mse_lam1_minmax"
KEYS = ("acc", "bal", "f1", "mcc", "auc")
MINORITY_LT_10 = {"soybean15", "cover10k", "sepsis"}  # train 최소 클래스 비율 < 10% (10-27 확인: 3.5% / 0.4% / 7.4%)


def load(path: Path):
    R = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["metrics"] is None:
            continue
        R[(r["dataset"], r["fold"], r["config"], r["seed"], r["split"], r["iterations"])] = r
    return R


def budget(ds_rows) -> int:
    return 100 if ds_rows["K"] <= 3 else 300


def main():
    out = Path(sys.argv[1])
    R = load(out / "runs.jsonl")
    folds = defaultdict(set)
    Kof = {}
    for (ds, f, c, s, sp, it), r in R.items():
        folds[ds].add(f)
        Kof[ds] = r["K"]
    lines = []
    P = lines.append

    def get(ds, f, c, s, sp, it, k):
        r = R.get((ds, f, c, s, sp, it))
        return None if r is None else r["metrics"][k]

    def mean_over(ds, c, sp, it, k):
        v = [get(ds, f, c, s, sp, it, k) for f in folds[ds] for s in (0, 1, 2)]
        v = [x for x in v if x is not None]
        return st.mean(v) if v else float("nan")

    for it_mode in ("budget", 500):
        for sel_key in ("mcc", "acc"):
            P(f"\n## iteration={it_mode}, 선택 지표={sel_key}\n")
            its = {ds: (100 if Kof[ds] <= 3 else 300) if it_mode == "budget" else 500 for ds in DATASETS}
            # global: 데이터셋별 val 평균의 평균
            gscore = {c: st.mean(mean_over(ds, c, "val", its[ds], sel_key) for ds in DATASETS) for c in CONFIGS}
            gbest = max(CONFIGS, key=lambda c: (gscore[c], -list(CONFIGS).index(c)))
            _, glam, gscale = CONFIGS[gbest]
            P(f"global 선택 조합: **{gbest}** (val {sel_key} 평균 {gscore[gbest]:.3f}, baseline {gscore[BASE]:.3f})\n")
            choices = defaultdict(lambda: defaultdict(int))

            def pick(ds, f, s, mode):
                if mode == "baseline":
                    return BASE
                if mode == "global":
                    return gbest
                if mode == "rule":
                    return f"{'inv' if ds in MINORITY_LT_10 else 'mse'}_lam{int(glam)}_{gscale}"
                sp = "val" if mode == "per_ds" else "test"
                cand = [c for c in CONFIGS if get(ds, f, c, s, sp, its[ds], sel_key) is not None]
                return max(cand, key=lambda c: (get(ds, f, c, s, sp, its[ds], sel_key), -list(CONFIGS).index(c)))

            modes = ("baseline", "global", "rule", "per_ds", "oracle")
            res = {m: {} for m in modes}
            for ds in DATASETS:
                for m in modes:
                    vals = defaultdict(list)
                    for f in folds[ds]:
                        for s in (0, 1, 2):
                            c = pick(ds, f, s, m)
                            if m == "per_ds":
                                choices[ds][c] += 1
                            for k in KEYS:
                                v = get(ds, f, c, s, "test", its[ds], k)
                                if v is not None:
                                    vals[k].append(v)
                    res[m][ds] = {k: st.mean(v) for k, v in vals.items()}
            P("| dataset | it | " + " | ".join(f"{m} acc/bal/MCC" for m in modes) + " |")
            P("|---|---|" + "---|" * len(modes))
            for ds in DATASETS:
                cells = [f"{res[m][ds]['acc']:.3f}/{res[m][ds]['bal']:.3f}/{res[m][ds]['mcc']:.3f}" for m in modes]
                P(f"| {ds} | {its[ds]} | " + " | ".join(cells) + " |")
            P("| **평균** | | " + " | ".join(
                "/".join(f"{st.mean(res[m][ds][k] for ds in DATASETS):.3f}" for k in ("acc", "bal", "mcc")) for m in modes) + " |")
            P("\n평균 macro F1 / AUC: " + ", ".join(
                f"{m} {st.mean(res[m][ds]['f1'] for ds in DATASETS):.3f}/{st.mean(res[m][ds]['auc'] for ds in DATASETS):.3f}" for m in modes))
            P("\nper_ds가 고른 조합(빈도): " + "; ".join(
                f"{ds}: " + ", ".join(f"{c}x{n}" for c, n in sorted(choices[ds].items(), key=lambda x: -x[1])[:3]) for ds in DATASETS))

    # 조합별 test 표(예산 iteration, fold·seed 평균) - 데이터셋별 상위 5개 + baseline
    P("\n## 데이터셋별 조합 test 순위 (예산 iteration, MCC 기준 상위 5 + baseline)\n")
    for ds in DATASETS:
        it = 100 if Kof[ds] <= 3 else 300
        sc = {c: {k: mean_over(ds, c, "test", it, k) for k in KEYS} for c in CONFIGS}
        top = sorted(CONFIGS, key=lambda c: -sc[c]["mcc"])[:5]
        if BASE not in top:
            top.append(BASE)
        P(f"**{ds}** (it {it}): " + "; ".join(
            f"{c}{'(base)' if c == BASE else ''} acc {sc[c]['acc']:.3f} bal {sc[c]['bal']:.3f} MCC {sc[c]['mcc']:.3f} AUC {sc[c]['auc']:.3f}"
            for c in top))
    # 요소별 효과(다른 요소 평균) - 데이터셋 평균 MCC
    P("\n## 요소별 주효과 (예산 iteration, test MCC, 다른 요소에 대해 평균)\n")
    for name, idx, levels in (("손실", 0, ("mse", "inv", "sqrt", "cap5")), ("λ", 1, (0.0, 1.0)), ("스케일링", 2, ("minmax", "clip5_95"))):
        row = []
        for lv in levels:
            cs = [c for c, v in CONFIGS.items() if v[idx] == lv]
            per_ds = {ds: st.mean(mean_over(ds, c, "test", 100 if Kof[ds] <= 3 else 300, "mcc") for c in cs) for ds in DATASETS}
            row.append(f"{lv}: 평균 {st.mean(per_ds.values()):.3f} (" + ", ".join(f"{ds[:6]} {v:.3f}" for ds, v in per_ds.items()) + ")")
        P(f"- {name}: " + " | ".join(row))
    # 구간 이탈
    ood = sorted({(k[0], k[2]) for k, r in R.items() if r["ood_first_it"] is not None})
    P(f"\n다항식 구간 이탈 (dataset, config): {ood if ood else '없음'}")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
