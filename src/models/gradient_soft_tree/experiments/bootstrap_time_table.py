"""bootstrap 시간 환산표(2026-10-20). GPU 본실험 iteration 기록에서 bootstrap 호출 수를 세고, bootstrap 1회 시간만 바꿨을 때의
학습 시간을 계산한다. bootstrap 속도는 라이브러리/하드웨어 문제라 알고리즘과 분리해서 보고한다(PAPER_ROADMAP '학습 시간 보고 방식').

- 입력: results/paper_gpu/sessions/*/measurements.jsonl (iteration마다 counts.bootstrap, counts.merge_bootstrap, t_total)
- 우리 small key 1회 시간은 측정값에서 회귀로 추정: iris/wine/breast명목(merge 수가 iteration마다 변함)에서
  t_total = a + b·single + c·merge → single 4.2~4.8s, merge 4.1~4.3s/호출. 데이터셋 고정효과 회귀는 single 4.37s
  (merge는 single과 공선성 때문에 따로 식별 안 됨) → single 4.37s, merge 4.2s 사용.
- bootstrap 외 시간 = 측정 t_total - (single·4.37 + merge·4.2). 다른 라이브러리로 바꿔도 이 부분은 그대로 둔다(하드웨어 차이 무시).
- 시나리오(호출 1회 시간 single / merge):
  small    : 4.37 / 4.2   (실측, RTX 4090, desilofhe small key)
  regular  : 1.52 / 1.54  (10-09 실측, 큐와 동시 실행이라 상한. 24GB GPU에 안 들어감 - 48GB급 필요)
  HEaaN    : 0.13 / 0.26  (HEaaN-ID3 논문 A6000 값, merge가 없다고 보고 ct 2개 = bootstrap 2회)
- soybean15는 세션 기록이 없어 EXPERIMENT_LOG 10-16 값(3주기 평균 single 45, merge 8.3, 442s/it)을 쓴다.

python -m models.gradient_soft_tree.experiments.bootstrap_time_table <results/paper_gpu/sessions> <out_dir>
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

SCENARIOS = {"small": (4.37, 4.2), "regular": (1.52, 1.54), "HEaaN": (0.13, 0.26)}
BUDGET = {"soybean15": 300}  # 나머지 100
CONVERGE_IT = {"iris": 20, "wine": 20, "breast_cancer": 20, "breast_cancer_nominal": 20, "diabetes": 30,
               "banknote": 90, "soybean15": 130}  # EXPERIMENT_LOG 10-17 수렴 지점
EXTRA = {"soybean15": (45.0, 8.3, 442.0)}  # (single, merge, t_iter) - 세션 기록 없음


def per_dataset(session_root: str) -> dict:
    acc: dict = {}
    for d in sorted(glob.glob(f"{session_root}/levelpacked_*")):
        ds = json.load(open(f"{d}/config.json"))["dataset"]
        for line in open(f"{d}/measurements.jsonl"):
            r = json.loads(line)
            if "counts" in r and r.get("bootstrap_key") == "small":
                acc.setdefault(ds, []).append((r["counts"].get("bootstrap", 0), r["counts"].get("merge_bootstrap", 0), r["t_total"]))
    out = {ds: tuple(np.mean(v, 0)) for ds, v in acc.items()}
    return out | {k: v for k, v in EXTRA.items() if k not in out}


def main():
    root, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    s0, m0 = SCENARIOS["small"]
    rows = []
    for ds, (single, merge, t) in per_dataset(root).items():
        other = t - (single * s0 + merge * m0)
        row = {"dataset": ds, "bootstrap_calls": round(single + merge, 1), "single": round(single, 1), "merge": round(merge, 1),
               "t_iter_s": round(t, 1), "non_bootstrap_s": round(other, 1), "bootstrap_share": round(1 - other / t, 3),
               "budget_it": BUDGET.get(ds, 100), "converge_it": CONVERGE_IT.get(ds)}
        for name, (s, m) in SCENARIOS.items():
            it = other + single * s + merge * m
            row[f"{name}_iter_s"] = round(it, 1)
            row[f"{name}_budget_h"] = round(it * row["budget_it"] / 3600, 2)
            if row["converge_it"]:
                row[f"{name}_converge_min"] = round(it * row["converge_it"] / 60, 1)
        rows.append(row)
    with open(out / "bootstrap_time_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'dataset':22s} {'calls':>5} {'boot%':>5} | it당 s: small/regular/HEaaN | 예산(h): small/regular/HEaaN | 수렴까지(분): small/regular/HEaaN")
    for r in rows:
        print(f"{r['dataset']:22s} {r['bootstrap_calls']:5.1f} {100 * r['bootstrap_share']:5.0f} | "
              + "/".join(f"{r[f'{n}_iter_s']:.0f}" for n in SCENARIOS) + " | "
              + "/".join(f"{r[f'{n}_budget_h']:.2f}" for n in SCENARIOS) + " | "
              + "/".join(f"{r.get(f'{n}_converge_min', float('nan')):.0f}" for n in SCENARIOS))


if __name__ == "__main__":
    main()
