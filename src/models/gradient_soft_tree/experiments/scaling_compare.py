"""[-1,1] 스케일링 방식 비교 (plaintext, CKKS 다항식 에뮬레이션, CPU). 2026-10-09.

방식(전부 client가 train으로 계산, train/test 모두 같은 변환 후 [-1,1] clip - clip 범위 밖 값은 경계값으로 묶임, 샘플은 제거하지 않음)
- minmax    : 지금 방식. [min, max] -> [-1, 1]
- clip1_99  : [1%, 99% 분위수] -> [-1, 1] (winsorized min-max, 선형)
- clip5_95  : [5%, 95% 분위수] -> [-1, 1]
- robust    : (x - 중앙값) / (2 * IQR). IQR = 0이면(이진/희소 feature) max-min으로 대체
- standard  : (x - 평균) / (3 * 표준편차) (참고용, 10-08 비교에서 범주형에 나빴음)
학습: 제안 방식(momentum 0.9, lr 1, kappa .25, leaf lr x K, 합성 clamp), depth 3(class>8이면 4), 예산 150(class<=3)/200.

python -m models.gradient_soft_tree.experiments.scaling_compare <out.jsonl>
"""

from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import evaluation_splits, one_hot_encode  # noqa: E402
from models.gradient_soft_tree.baseline.reference import DEFAULT_POLY_SPEC, domain_report, init_params, predict  # noqa: E402
from models.gradient_soft_tree.levelpacked import tree as T  # noqa: E402

DATASETS = ["iris", "wine", "breast_cancer", "breast_cancer_nominal", "diabetes", "banknote", "soybean15", "digits",
            "cover10k", "sepsis"]
SCALERS = ["minmax", "clip1_99", "clip5_95", "robust", "standard"]
SAFE = T.OptConfig("mom", 1.0, 0.9, 0.25, True, leaf_k_scale=True)
CHECKS = [30, 50, 100, 150, 200]


def fit_transform(kind: str, Xtr: np.ndarray, Xte: np.ndarray):
    if kind in ("minmax", "clip1_99", "clip5_95"):
        lo_q, hi_q = {"minmax": (0, 100), "clip1_99": (1, 99), "clip5_95": (5, 95)}[kind]
        lo, hi = np.percentile(Xtr, lo_q, axis=0), np.percentile(Xtr, hi_q, axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        f = lambda A: 2 * (A - lo) / span - 1
    elif kind == "robust":
        med = np.median(Xtr, axis=0)
        iqr = np.percentile(Xtr, 75, axis=0) - np.percentile(Xtr, 25, axis=0)
        rng = Xtr.max(0) - Xtr.min(0)
        scale = np.where(iqr > 0, 2 * iqr, np.where(rng > 0, rng, 1.0))
        f = lambda A: (A - med) / scale
    else:
        mu, sd = Xtr.mean(0), Xtr.std(0)
        f = lambda A: (A - mu) / (3 * np.where(sd > 0, sd, 1.0))
    return np.clip(f(Xtr), -1, 1), np.clip(f(Xte), -1, 1)


def job(args):
    ds, fold, kind = args
    Xr, Xter, y, yte = evaluation_splits(ds)[fold]
    X, Xte = fit_transform(kind, Xr, Xter)
    K = int(max(y.max(), yte.max()) + 1)
    depth = 3 if K <= 8 else 4
    T_budget = 150 if K <= 3 else 200
    hist = T.emulate(X, one_hot_encode(y, K), depth, T.default_spec(), SAFE, max(CHECKS),
                     init_params(X.shape[1], K, depth, 0))
    acc = {}
    for it in CHECKS:
        with np.errstate(all="ignore"):
            acc[it] = float((predict(Xte, {**hist[it - 1], "steepness": 8.0, "depth": depth}, DEFAULT_POLY_SPEC) == yte).mean())
    # 해상도 지표: train에서 feature별로 clip된 비율, 값이 몰린 정도(표준편차 평균)
    clipped = float(np.mean(np.abs(X) >= 1 - 1e-9))
    return {"dataset": ds, "fold": fold, "scaler": kind, "budget": T_budget, "acc": acc, "clipped_frac": clipped,
            "feature_std": float(X.std(0).mean()),
            "domain_ok": not domain_report(X, {**hist[T_budget - 1], "steepness": 8.0, "depth": depth})["n_samples_gate_out_of_range"]}


def main() -> None:
    out = Path(sys.argv[1])
    jobs = [(ds, f, k) for ds in DATASETS for f in range(len(evaluation_splits(ds))) for k in SCALERS]
    with Pool(int(os.environ.get("N_WORKERS", "12"))) as pool:
        rows = pool.map(job, jobs)
    out.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"{'dataset':22s} " + " ".join(f"{k:>14s}" for k in SCALERS) + "   (예산 iteration의 test 정확도, 3-fold 평균±표준편차)")
    for ds in DATASETS:
        cells = []
        for k in SCALERS:
            rs = [r for r in rows if r["dataset"] == ds and r["scaler"] == k]
            a = [r["acc"][r["budget"]] for r in rs]
            cells.append(f"{np.mean(a):.3f}±{np.std(a):.3f}" if len(a) > 1 else f"{a[0]:.3f}")
        print(f"{ds:22s} " + " ".join(f"{c:>14s}" for c in cells))
    print("\n평균(10개 데이터셋)   " + " ".join(
        f"{np.mean([np.mean([r['acc'][r['budget']] for r in rows if r['dataset'] == ds and r['scaler'] == k]) for ds in DATASETS]):>14.3f}"
        for k in SCALERS))
    print("clip된 값 비율(train) " + " ".join(f"{np.mean([r['clipped_frac'] for r in rows if r['scaler'] == k]):>14.3f}" for k in SCALERS))
    print("feature 표준편차 평균  " + " ".join(f"{np.mean([r['feature_std'] for r in rows if r['scaler'] == k]):>14.3f}" for k in SCALERS))
    print("구간 이탈 run 수       " + " ".join(f"{sum(not r['domain_ok'] for r in rows if r['scaler'] == k):>14d}" for k in SCALERS))


if __name__ == "__main__":
    main()
