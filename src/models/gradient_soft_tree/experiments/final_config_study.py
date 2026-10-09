"""최종 방법 설정 결정(2026-10-09): balance penalty λ ∈ {0, 1, 3, 10} x multi-start(R=8 평균 앙상블) 유무, 3-fold.

- 단일 모델 = seed 0~7 각각의 정확도 평균, 앙상블 = seed 0~7의 예측 확률 평균
- λ 선택: fold train을 fit/val로 다시 나눈 inner validation 정확도(전 데이터셋 평균) - test 미사용
- 보고: fold train 전체로 학습한 test 정확도(3-fold 평균, holdout 데이터셋은 1개)

python -m models.gradient_soft_tree.experiments.final_config_study <out.jsonl>
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

DATASETS = ["iris", "wine", "breast_cancer", "breast_cancer_nominal", "diabetes", "banknote", "soybean15", "digits",
            "cover10k", "sepsis"]
LAMBDAS = [0.0, 1.0, 3.0, 10.0]
SEEDS = range(8)


def job(args):
    ds, fold, lam, seed, inner = args
    r = train(ds, seed, Method(f"lam={lam}", balance=lam), fold=fold, inner=inner)
    return {k: r[k] for k in ("dataset", "fold", "inner", "seed", "L", "test_acc", "yhat_te")} | {"lam": lam}


def summarize(rows):
    def acc_table(inner):
        out = {}
        for ds in DATASETS:
            for lam in LAMBDAS:
                single, ens = [], []
                folds = sorted({r["fold"] for r in rows if r["dataset"] == ds})
                for f in folds:
                    rs = [r for r in rows if r["dataset"] == ds and r["fold"] == f and r["lam"] == lam and r["inner"] == inner]
                    if not rs:
                        continue
                    yte = _data(ds, f, inner)[3]
                    single.append(np.mean([r["test_acc"] for r in rs]))
                    ens.append(float((np.mean([np.array(r["yhat_te"]) for r in rs], 0).argmax(1) == yte).mean()))
                out[(ds, lam)] = (single, ens)
        return out

    val, test = acc_table(True), acc_table(False)
    print("== λ 선택(inner validation 정확도, 데이터셋 평균) ==")
    for lam in LAMBDAS:
        print(f"λ={lam:<5} 단일 {np.mean([np.mean(val[(d, lam)][0]) for d in DATASETS]):.4f}  "
              f"앙상블 {np.mean([np.mean(val[(d, lam)][1]) for d in DATASETS]):.4f}")
    best_s = max(LAMBDAS, key=lambda l: np.mean([np.mean(val[(d, l)][0]) for d in DATASETS]))
    best_e = max(LAMBDAS, key=lambda l: np.mean([np.mean(val[(d, l)][1]) for d in DATASETS]))
    print(f"-> 선택: 단일 모델 λ={best_s}, 앙상블 λ={best_e}\n")
    print("== test 정확도 (3-fold 평균; 단일 = seed 8개 평균, 앙상블 = 8개 평균 예측) ==")
    hdr = " ".join(f"{('λ'+str(l)):>8s}" for l in LAMBDAS)
    print(f"{'dataset':22s} 단일[{hdr}] | 앙상블[{hdr}]")
    for ds in DATASETS:
        s = " ".join(f"{np.mean(test[(ds, l)][0]):8.3f}" for l in LAMBDAS)
        e = " ".join(f"{np.mean(test[(ds, l)][1]):8.3f}" for l in LAMBDAS)
        print(f"{ds:22s}     [{s}] |       [{e}]")
    print(f"{'평균':22s}     [" + " ".join(f"{np.mean([np.mean(test[(d, l)][0]) for d in DATASETS]):8.3f}" for l in LAMBDAS)
          + "] |       [" + " ".join(f"{np.mean([np.mean(test[(d, l)][1]) for d in DATASETS]):8.3f}" for l in LAMBDAS) + "]")


def main():
    out = Path(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "summary":
        summarize([json.loads(l) for l in open(out)])
        return
    jobs = [(ds, f, lam, s, inner) for ds in DATASETS for f in range(len(evaluation_splits(ds)))
            for lam in LAMBDAS for s in SEEDS for inner in (True, False)]
    print(f"[final] {len(jobs)} runs", flush=True)
    with Pool(int(os.environ.get("N_WORKERS", "12"))) as pool, open(out, "w") as fh:
        rows = []
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=2), 1):
            fh.write(json.dumps(r) + "\n")
            rows.append(r)
            if i % 300 == 0:
                print(f"[final] {i}/{len(jobs)}", flush=True)
    summarize(rows)


if __name__ == "__main__":
    main()
