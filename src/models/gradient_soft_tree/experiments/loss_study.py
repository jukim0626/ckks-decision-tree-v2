"""loss 비교(2026-10-09): MSE(지금) vs 가중 MSE(클래스 빈도 역수) vs expected CE.
현재 방법(balance λ=1, const lr, depth 3/4, 예산 150/200), 3-fold(holdout 1개), seed 4개(단일 평균 + 4개 앙상블).
loss마다 lr 배율 {0.5,1,2}를 inner validation(정확도+balanced acc 평균)으로 골라 비교. 지표: accuracy, balanced accuracy, macro F1.

python -m models.gradient_soft_tree.experiments.loss_study <out.jsonl>
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import evaluation_splits  # noqa: E402
from models.gradient_soft_tree.experiments.global_opt_study import Method, _data, train  # noqa: E402

DATASETS = ["iris", "wine", "breast_cancer", "breast_cancer_nominal", "diabetes", "banknote", "soybean15", "digits",
            "cover10k", "sepsis"]
LOSSES = ["mse", "wmse", "ece"]
LR_MULTS = [0.5, 1.0, 2.0]
SEEDS = range(4)


def job(args):
    ds, fold, loss, mult, seed, inner = args
    r = train(ds, seed, Method(f"{loss}x{mult}", balance=1.0, loss=loss, lr_mult=mult), fold=fold, inner=inner)
    return {"dataset": ds, "fold": fold, "loss": loss, "mult": mult, "seed": seed, "inner": inner, "yhat_te": r["yhat_te"]}


def metrics(yhat, yte):
    p = np.asarray(yhat).argmax(1)
    return (float((p == yte).mean()), float(balanced_accuracy_score(yte, p)), float(f1_score(yte, p, average="macro")))


def main():
    out = Path(sys.argv[1])
    jobs = [(ds, f, l, m, s, inner) for ds in DATASETS for f in range(len(evaluation_splits(ds))) for l in LOSSES
            for m in LR_MULTS for s in SEEDS for inner in (True, False)]
    print(f"[loss] {len(jobs)} runs", flush=True)
    with Pool(int(os.environ.get("N_WORKERS", "12"))) as pool:
        rows = pool.map(job, jobs, chunksize=2)
    out.write_text("\n".join(json.dumps({k: v for k, v in r.items() if k != "yhat_te"} | {"yhat_te": r["yhat_te"]}) for r in rows))

    def score(ds, loss, mult, inner, ensemble):
        res = []
        for f in range(len(evaluation_splits(ds))):
            yte = _data(ds, f, inner)[3]
            rs = [r for r in rows if r["dataset"] == ds and r["fold"] == f and r["loss"] == loss and r["mult"] == mult and r["inner"] == inner]
            if ensemble:
                res.append(metrics(np.mean([np.array(r["yhat_te"]) for r in rs], 0), yte))
            else:
                res.append(tuple(np.mean([metrics(r["yhat_te"], yte) for r in rs], 0)))
        return np.mean(res, 0)

    best = {}
    for loss in LOSSES:
        val = {m: np.mean([np.mean(score(ds, loss, m, True, False)[:2]) for ds in DATASETS]) for m in LR_MULTS}
        best[loss] = max(val, key=val.get)
        print(f"{loss:5s} lr 배율 val: " + " ".join(f"x{m}={v:.4f}" for m, v in val.items()) + f" -> x{best[loss]}")
    for ens in (False, True):
        print(f"\n== test ({'앙상블 4개' if ens else '단일 모델 평균'}): accuracy / balanced acc / macro F1 ==")
        print(f"{'dataset':22s} " + " ".join(f"{l + ' x' + str(best[l]):>22s}" for l in LOSSES))
        tot = {l: [] for l in LOSSES}
        for ds in DATASETS:
            cells = []
            for l in LOSSES:
                a = score(ds, l, best[l], False, ens)
                tot[l].append(a)
                cells.append(f"{a[0]:.3f}/{a[1]:.3f}/{a[2]:.3f}")
            print(f"{ds:22s} " + " ".join(f"{c:>22s}" for c in cells))
        print(f"{'평균':22s} " + " ".join(f"{'/'.join(f'{x:.3f}' for x in np.mean(tot[l], 0)):>22s}" for l in LOSSES))


if __name__ == "__main__":
    main()
