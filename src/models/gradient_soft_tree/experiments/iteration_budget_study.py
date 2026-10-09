"""iteration 예산 근거(2026-10-09, 논문 Phase: 학습 비용). CKKS 다항식 에뮬레이션(plaintext, CPU).

질문: 150/200 iteration이 정말 필요한가? 정확도가 언제 거의 다 올라오는가(time-to-accuracy)?
현재 방법(momentum 0.9, η=1, 노드 lr 0.25F, leaf lr K, 합성 clamp, balance λ=1, multi-start R=free_models)을
데이터셋 x fold마다 MAX_T iteration 돌리면서 매 STEP iteration마다 train loss / test accuracy / balanced acc를 기록한다.

python -m models.gradient_soft_tree.experiments.iteration_budget_study <out_dir>
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import evaluation_splits, fit_scaler, one_hot_encode, scale_features  # noqa: E402
from models.gradient_soft_tree.baseline.reference import (  # noqa: E402
    DEFAULT_POLY_SPEC, compute_gradients, init_params, predict_proba,
)
from models.gradient_soft_tree.levelpacked import tree as T  # noqa: E402
from models.gradient_soft_tree.levelpacked.layout import free_models  # noqa: E402

DATASETS = ["iris", "wine", "breast_cancer", "breast_cancer_nominal", "diabetes", "banknote",
            "soybean15", "digits", "cover10k", "sepsis"]
MAX_T = 300
STEP = 10
OPT = T.OptConfig("mom", 1.0, 0.9, 0.25, True, leaf_k_scale=True, balance=1.0)


def balanced_acc(y, p, K):
    return float(np.mean([(p[y == c] == c).mean() for c in range(K) if (y == c).any()]))


def job(args):
    ds, fold = args
    Xtr_raw, Xte_raw, ytr, yte = evaluation_splits(ds)[fold]
    sc = fit_scaler(Xtr_raw)
    X, Xte = scale_features(sc, Xtr_raw), scale_features(sc, Xte_raw)
    K = int(max(ytr.max(), yte.max()) + 1)
    depth = 4 if K > 8 else 3
    Y = one_hot_encode(ytr, K)
    R = free_models(len(ytr), X.shape[1], K, depth)
    hists = []
    with np.errstate(all="ignore"):
        for r in range(R):
            hists.append(T.emulate(X, Y, depth, T.default_spec(), OPT, MAX_T, init_params(X.shape[1], K, depth, 1000 * r)))
    rows = []
    for t in range(STEP, MAX_T + 1, STEP):
        ps = [{**h[t - 1], "steepness": 8.0, "depth": depth} for h in hists]
        with np.errstate(all="ignore"):
            pr_single = predict_proba(Xte, ps[0], DEFAULT_POLY_SPEC)
            pr_ens = np.mean([predict_proba(Xte, p, DEFAULT_POLY_SPEC) for p in ps], 0)
            L = float(compute_gradients(X, Y, hists[0][t - 1], depth, approx=DEFAULT_POLY_SPEC)[1])
        a1, ae = pr_single.argmax(1), pr_ens.argmax(1)
        rows.append({"dataset": ds, "fold": fold, "t": t, "R": R, "depth": depth, "train_loss": L,
                     "acc": float((a1 == yte).mean()), "bal": balanced_acc(yte, a1, K),
                     "acc_ens": float((ae == yte).mean()), "bal_ens": balanced_acc(yte, ae, K)})
    print(f"[budget] {ds} fold {fold} done (R={R})", flush=True)
    return rows


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(ds, f) for ds in DATASETS for f in range(len(evaluation_splits(ds)))]
    with Pool(int(os.environ.get("N_WORKERS", "12"))) as pool:
        res = pool.map(job, jobs, chunksize=1)
    with open(out / "curves.jsonl", "w") as f:
        for rows in res:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
