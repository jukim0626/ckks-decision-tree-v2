"""lr schedule 비교(2026-10-09). 장시간 학습 불안정의 원인이 노드 step 과대로 확인되어, 예산 T가 고정된 HE 상황에 맞는
schedule을 고른다. 모든 조건에 balance λ=1, depth 3(K>8이면 4), 3-fold(holdout 1개), seed 4개(단일 평균 + 4개 앙상블).
예산 T = 50/150/300 각각에서 비교(예산이 바뀌어도 잘 되는지).

python -m models.gradient_soft_tree.experiments.lr_schedule_study <out.jsonl>
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import evaluation_splits  # noqa: E402
from models.gradient_soft_tree.experiments.global_opt_study import Method, _data, train  # noqa: E402

DATASETS = ["iris", "wine", "breast_cancer", "breast_cancer_nominal", "diabetes", "banknote", "soybean15", "digits", "cover10k"]
BUDGETS = [50, 150, 300]
SCHEDS = {
    "const(지금)": Method("const", balance=1.0),
    "linear": Method("linear", balance=1.0, lr_sched="linear"),
    "cosine": Method("cosine", balance=1.0, lr_sched="cosine"),
    "sqrt": Method("sqrt", balance=1.0, lr_sched="sqrt"),
    "node_cap2": Method("cap2", balance=1.0, node_cap=2.0),
    "step_clip0.1": Method("stepclip", balance=1.0, step_clip=0.1),
}
SEEDS = range(4)


def job(args):
    ds, fold, name, T, seed = args
    r = train(ds, seed, SCHEDS[name], T=T, fold=fold)
    return {"dataset": ds, "fold": fold, "sched": name, "T": T, "seed": seed, "L": r["L"], "test_acc": r["test_acc"],
            "yhat_te": r["yhat_te"], "pre_clamp_max": r["pre_clamp_max"]}


def main():
    out = Path(sys.argv[1])
    jobs = [(ds, f, n, T, s) for ds in DATASETS for f in range(len(evaluation_splits(ds))) for n in SCHEDS for T in BUDGETS for s in SEEDS]
    print(f"[lr] {len(jobs)} runs", flush=True)
    with Pool(int(os.environ.get("N_WORKERS", "12"))) as pool:
        rows = pool.map(job, jobs, chunksize=2)
    out.write_text("\n".join(json.dumps(r) for r in rows))
    for T in BUDGETS:
        print(f"\n== 예산 T={T}: test 정확도 (3-fold 평균; 단일 = seed 4개 평균 / 앙상블 = 4개 평균 예측) ==")
        print(f"{'dataset':22s} " + " ".join(f"{n:>17s}" for n in SCHEDS))
        tot = {n: ([], []) for n in SCHEDS}
        for ds in DATASETS:
            cells = []
            for n in SCHEDS:
                singles, ens = [], []
                for f in range(len(evaluation_splits(ds))):
                    rs = [r for r in rows if r["dataset"] == ds and r["fold"] == f and r["sched"] == n and r["T"] == T]
                    yte = _data(ds, f)[3]
                    singles.append(np.mean([r["test_acc"] for r in rs]))
                    ens.append(float((np.mean([np.array(r["yhat_te"]) for r in rs], 0).argmax(1) == yte).mean()))
                tot[n][0].append(np.mean(singles)); tot[n][1].append(np.mean(ens))
                cells.append(f"{np.mean(singles):.3f}/{np.mean(ens):.3f}")
            print(f"{ds:22s} " + " ".join(f"{c:>17s}" for c in cells))
        print(f"{'평균':22s} " + " ".join(f"{np.mean(tot[n][0]):.3f}/{np.mean(tot[n][1]):.3f}".rjust(17) for n in SCHEDS))


if __name__ == "__main__":
    main()
