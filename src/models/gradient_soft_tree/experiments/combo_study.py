"""개선 후보 조합 비교(2026-10-27). CKKS 다항식 에뮬레이션(plaintext, CPU). 전부 CKKS에서 그대로 구현 가능한 조합만.

조합 = 손실 x balance λ x 스케일링 (16개):
  손실     mse | inv(클래스 빈도 역수 가중) | sqrt(역수의 제곱근) | cap5(역수, 평균 1 기준 상한 5)  - 10-27 imbalance_study
  λ        0 | 1                                                                         - 10-09 ablation(soybean은 λ=1이 -7.5%p)
  스케일링 minmax(본실험) | clip5_95(5/95% 분위수 -> [-1,1] 후 clip)                         - 10-09 scaling_compare(cover +12%p)
나머지는 본실험과 같음(momentum, 노드 lr 0.25F, leaf lr xK, clamp, R=free_models(전체 train 기준)).
iteration 30/50/100/200/300/500에서 평가. 판정은 argmax와 prior 보정(학습 후, 비용 없음) 둘 다 기록.

**test로 조합을 고르지 않는다**: 각 (fold, seed)마다 train을 다시 75/25(층화)로 나눠 같은 조합을 inner-train에 학습하고
inner-val 점수를 따로 기록한다(split="val"). 선택은 val로, 보고는 전체 train 학습 -> test(split="test").

cd src && python -m models.gradient_soft_tree.experiments.combo_study ../results/combo_study [n_proc]
(이미 기록된 (dataset, fold, config, seed, split)는 건너뛴다.)
"""
from __future__ import annotations

import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import evaluation_splits, fit_scaler, one_hot_encode, scale_features  # noqa: E402
from models.gradient_soft_tree.baseline.reference import DEFAULT_POLY_SPEC, domain_report, init_params, predict_proba  # noqa: E402
from models.gradient_soft_tree.experiments.ablation_study import DATASETS, FULL  # noqa: E402
from models.gradient_soft_tree.experiments.imbalance_study import class_weights, emulate_w, metrics  # noqa: E402
from models.gradient_soft_tree.experiments.scaling_compare import fit_transform  # noqa: E402
from models.gradient_soft_tree.levelpacked import tree as T  # noqa: E402
from models.gradient_soft_tree.levelpacked.layout import free_models  # noqa: E402
from dataclasses import replace  # noqa: E402

LOSSES = {"mse": None, "inv": "inv", "sqrt": "sqrt", "cap5": "inv_cap5"}
LAMBDAS = (0.0, 1.0)
SCALINGS = ("minmax", "clip5_95")
CONFIGS = {f"{l}_lam{int(lam)}_{s}": (l, lam, s) for l, lam, s in itertools.product(LOSSES, LAMBDAS, SCALINGS)}
# 10-27 2차: clip5_95는 digits에서 5%==95% 분위수인 feature 14개를 상수로 만든다 -> 그런 feature만 min-max로(fallback).
# 1차 결과에서 λ=1이 평균 우세라 λ=1만.
CONFIGS.update({f"{l}_lam1_clip5_95fb": (l, 1.0, "clip5_95fb") for l in LOSSES})
SEEDS = (0, 1, 2)
CHECKPOINTS = (30, 50, 100, 200, 300, 500)


def scale(kind: str, Xtr_raw: np.ndarray, Xev_raw: np.ndarray):
    if kind == "minmax":  # 본실험과 같은 경로(MinMaxScaler + clip)
        sc = fit_scaler(Xtr_raw)
        return scale_features(sc, Xtr_raw), scale_features(sc, Xev_raw)
    if kind == "clip5_95fb":  # 5/95% 분위수, 단 두 분위수가 같은 feature는 min/max로
        lo, hi = np.percentile(Xtr_raw, 5, axis=0), np.percentile(Xtr_raw, 95, axis=0)
        mn, mx = Xtr_raw.min(0), Xtr_raw.max(0)
        deg = hi <= lo
        lo, hi = np.where(deg, mn, lo), np.where(deg, mx, hi)
        span = np.where(hi > lo, hi - lo, 1.0)
        f = lambda A: np.clip(2 * (A - lo) / span - 1, -1, 1)
        return f(Xtr_raw), f(Xev_raw)
    return fit_transform(kind, Xtr_raw, Xev_raw)


def inner_split(X, y, seed: int = 0):
    try:
        return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    except ValueError:  # 클래스 샘플이 너무 적어 층화가 안 되면 일반 분할
        return train_test_split(X, y, test_size=0.25, random_state=seed)


def job(args):
    ds, fold, name, seed, split = args
    loss, lam, scaling = CONFIGS[name]
    Xtr_raw, Xte_raw, ytr, yte = evaluation_splits(ds)[fold]
    K = int(max(ytr.max(), yte.max()) + 1)
    depth = 4 if K > 8 else 3
    R = free_models(len(ytr), Xtr_raw.shape[1], K, depth)
    if split == "val":
        Xtr_raw, Xte_raw, ytr, yte = inner_split(Xtr_raw, ytr)
    X, Xev = scale(scaling, Xtr_raw, Xte_raw)
    prior = np.bincount(ytr, minlength=K) / len(ytr)
    w = class_weights(ytr, K, LOSSES[loss])
    opt = replace(FULL, balance=lam)
    hists, ood_at = [], None
    with np.errstate(all="ignore"):
        for r in range(R):
            init = init_params(X.shape[1], K, depth, 1000 * r + 100003 * seed)
            h = emulate_w(X, one_hot_encode(ytr, K), depth, T.default_spec(), opt, max(CHECKPOINTS), init, w)
            for i in range(9, len(h), 10):
                d = domain_report(X, {**h[i], "steepness": 8.0, "depth": depth})
                if (d["n_samples_gate_out_of_range"] > 0 or d["softmax_out_of_range"]
                        or not all(np.isfinite(v).all() for v in h[i].values())):
                    ood_at = i + 1 if ood_at is None else min(ood_at, i + 1)
                    break
            hists.append(h)
        rows = []
        for it in CHECKPOINTS:
            probs = [predict_proba(Xev, {**h[it - 1], "steepness": 8.0, "depth": depth}, DEFAULT_POLY_SPEC) for h in hists]
            ok = [p for p in probs if np.isfinite(p).all()]
            m = metrics(yte, np.mean(ok, 0), K, prior) if ok else None
            rows.append({"dataset": ds, "fold": fold, "config": name, "loss": loss, "lam": lam, "scaling": scaling,
                         "seed": seed, "split": split, "R": R, "K": K, "iterations": it, "n_eval": len(yte),
                         "ood_first_it": ood_at, "metrics": m})
    return rows


def main():
    out = Path(sys.argv[1])
    n_proc = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    out.mkdir(parents=True, exist_ok=True)
    path = out / "runs.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["dataset"], r["fold"], r["config"], r["seed"], r["split"]))
    jobs = [(ds, f, name, s, sp) for ds in DATASETS for f in range(len(evaluation_splits(ds)))
            for name in CONFIGS for s in SEEDS for sp in ("val", "test") if (ds, f, name, s, sp) not in done]
    jobs.sort(key=lambda j: j[0] in ("sepsis", "cover10k", "digits"))  # 오래 걸리는 데이터셋을 뒤로
    print(f"{len(jobs)} jobs ({len(done)} done)", flush=True)
    with Pool(n_proc) as pool, open(path, "a") as f:
        for i, rows in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i + 1}/{len(jobs)} {rows[-1]['dataset']} {rows[-1]['config']}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
