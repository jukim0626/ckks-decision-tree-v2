"""depth 선택 비교(2026-10-09): 클래스 수 <= leaf 수를 만족하는 최소 depth vs 더 깊은 depth.
최종 설정(제안 방식 + balance λ=1), seed 4개 평균, 3-fold(holdout 데이터셋은 1개), 예산 150(class<=3)/200.
선택은 inner validation(fold train 안에서 다시 나눈 75/25), 보고는 test.

python -m models.gradient_soft_tree.experiments.depth_study <out.jsonl>
"""
from __future__ import annotations

import json
import os
import sys
from math import ceil, log2
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import evaluation_splits  # noqa: E402
from models.gradient_soft_tree.experiments.global_opt_study import Method, _data, train  # noqa: E402

DATASETS = ["iris", "wine", "breast_cancer", "breast_cancer_nominal", "diabetes", "banknote", "soybean15", "digits",
            "cover10k", "sepsis"]
M = Method("lam1", balance=1.0)


def depths_for(K: int) -> list[int]:
    d0 = max(1, ceil(log2(K)))   # leaf 수 2^d >= K 인 최소 depth
    return list(range(d0, max(d0, 4) + 1)) if K <= 8 else [d0, d0 + 1]


def job(args):
    ds, fold, depth, seed, inner = args
    r = train(ds, seed, M, fold=fold, inner=inner, depth=depth)
    return {k: r[k] for k in ("dataset", "fold", "inner", "seed", "depth", "L", "test_acc")}


def main():
    out = Path(sys.argv[1])
    jobs = []
    for ds in DATASETS:
        K = _data(ds)[4]
        for f in range(len(evaluation_splits(ds))):
            for d in depths_for(K):
                for s in range(4):
                    for inner in (True, False):
                        jobs.append((ds, f, d, s, inner))
    with Pool(int(os.environ.get("N_WORKERS", "12"))) as pool:
        rows = pool.map(job, jobs, chunksize=2)
    out.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"{'dataset':22s} {'K':>2s}  depth별 test 정확도(3-fold, seed 4개 평균)        val로 고른 depth")
    for ds in DATASETS:
        K = _data(ds)[4]
        cells, vals = [], {}
        for d in depths_for(K):
            t = [r["test_acc"] for r in rows if r["dataset"] == ds and r["depth"] == d and not r["inner"]]
            v = [r["test_acc"] for r in rows if r["dataset"] == ds and r["depth"] == d and r["inner"]]
            vals[d] = np.mean(v)
            cells.append(f"d{d}={np.mean(t):.3f}")
        print(f"{ds:22s} {K:>2d}  " + "  ".join(cells) + f"   -> d{max(vals, key=vals.get)} (최소 d{depths_for(K)[0]})")


if __name__ == "__main__":
    main()
