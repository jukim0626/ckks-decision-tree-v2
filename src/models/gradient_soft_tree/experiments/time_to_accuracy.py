"""time-to-accuracy 표(2026-10-27): GPU 본실험 중간 평가(30/50/100, 다중 클래스 100/200/300)의 암호 추론 정확도와
그 시점까지의 순수 학습 시간(train_iter_sum_s, 중간 평가 제외)을 fold 평균으로 정리한다.

cd src && python -m models.gradient_soft_tree.experiments.time_to_accuracy ../results/paper_gpu/final_runs.jsonl ../results/time_to_accuracy
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    runs = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    by = defaultdict(lambda: defaultdict(list))  # dataset -> iterations -> [run]
    for r in runs:
        by[r["dataset"]][r["iterations"]].append(r)
    rows = []
    for ds, its in by.items():
        budget = max(its)
        final = {r["fold"]: r for r in its[budget]}
        for it in sorted(its):
            rs = its[it]
            acc = [r["predict"]["encrypted_test_acc"] for r in rs]
            hours = [r["times"]["train_iter_sum_s"] / 3600 for r in rs]
            frac = [r["times"]["train_iter_sum_s"] / final[r["fold"]]["times"]["train_iter_sum_s"] for r in rs if r["fold"] in final]
            gap = [final[r["fold"]]["predict"]["encrypted_test_acc"] - r["predict"]["encrypted_test_acc"] for r in rs if r["fold"] in final]
            rows.append({"dataset": ds, "iterations": it, "n_folds": len(rs), "folds": sorted(r["fold"] for r in rs),
                         "enc_acc_mean": sum(acc) / len(acc), "enc_acc_per_fold": acc,
                         "train_hours_mean": sum(hours) / len(hours), "time_frac_of_budget": sum(frac) / len(frac),
                         "acc_gap_to_budget_mean": sum(gap) / len(gap), "acc_gap_to_budget_max": max(gap)})
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.json").write_text(json.dumps(rows, indent=1))
    with open(out / "table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k not in ("enc_acc_per_fold", "folds")], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['dataset']:22s} it={r['iterations']:3d} folds={r['folds']} acc={r['enc_acc_mean']:.3f} "
              f"time={r['train_hours_mean']:.2f}h ({r['time_frac_of_budget']:.0%}) gap_to_budget mean={r['acc_gap_to_budget_mean']:+.3f} "
              f"max={r['acc_gap_to_budget_max']:+.3f}")


if __name__ == "__main__":
    main()
