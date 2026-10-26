"""클래스 가중 손실(wmse) 비교(2026-10-27). CKKS 다항식 에뮬레이션(plaintext, CPU) - GPU 본실험과 같은 분할/초기값/R.

10-26 sepsis 진단에서 본실험 설정(mse + balance λ=1)은 불균형 데이터에서 소수 클래스를 argmax로 못 고르고,
클래스 빈도 역수 가중 MSE(wmse)가 이를 고친다는 걸 확인했다. 이걸 10개 데이터셋 전 fold로 확장한다.
  mse        : 본실험(λ=1, multi-start R=free_models)
  wmse       : 클래스 빈도 역수 가중(평균 1)
  wmse_sqrt  : 역수의 제곱근 가중(평균 1)
seed 0은 본실험과 같은 초기값(1000*r), seed 1/2는 추가 초기값. 30/50/100/200/300 iteration마다 test 평가.
지표: acc, balanced acc, macro F1, MCC, AUC(이진: 소수 클래스, 다중: macro OvR), 혼동행렬, prior 보정(p/prior argmax) 판정,
학습 중 다항식 구간 이탈/비유한값 여부.

cd src && python -m models.gradient_soft_tree.experiments.imbalance_study ../results/imbalance_study [n_proc]
(이미 기록된 (dataset, fold, config, seed)는 건너뛴다 - 중간에 끊겨도 다시 실행하면 이어서 한다.)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import evaluation_splits, fit_scaler, one_hot_encode, scale_features  # noqa: E402
from models.gradient_soft_tree.baseline.reference import (DEFAULT_POLY_SPEC, PolySpec, compute_gradients,  # noqa: E402
                                                          domain_report, init_params, predict_proba)
from models.gradient_soft_tree.experiments.ablation_study import DATASETS, FULL  # noqa: E402
from models.gradient_soft_tree.levelpacked import he_ops as H  # noqa: E402
from models.gradient_soft_tree.levelpacked import tree as T  # noqa: E402
from models.gradient_soft_tree.levelpacked.layout import free_models  # noqa: E402

CONFIGS = {"mse": None, "wmse": "inv", "wmse_sqrt": "sqrt"}
# 10-27 추가: cover10k에서 역수 가중(최대 38배)이 극소수 클래스를 과하게 예측 -> 가중 상한 비교(불균형 3개 데이터셋만)
EXTRA_CONFIGS = {"wmse_cap5": "inv_cap5", "wmse_cap10": "inv_cap10"}
EXTRA_DATASETS = ("soybean15", "cover10k", "sepsis")
SEEDS = (0, 1, 2)
CHECKPOINTS = (30, 50, 100, 200, 300)


def class_weights(ytr: np.ndarray, K: int, kind: str | None) -> np.ndarray | None:
    """샘플별 가중치(평균 1). kind=None이면 가중 없음."""
    if kind is None:
        return None
    prior = np.bincount(ytr, minlength=K) / len(ytr)
    cw = np.where(prior > 0, 1 / np.maximum(prior, 1e-12), 0.0)
    if kind == "sqrt":
        cw = np.sqrt(cw)
    w = cw[ytr]
    w = w / w.mean()
    if kind.startswith("inv_cap"):  # 평균 1로 맞춘 뒤 상한을 자르고 다시 평균 1로
        w = np.minimum(w, float(kind[len("inv_cap"):]))
        w = w / w.mean()
    return w


def emulate_w(X, Y, depth, spec, opt, iterations, params0, sample_weight):
    """levelpacked.tree.emulate와 같은 규칙(momentum + 노드/leaf lr + clamp)에 sample_weight만 추가한 복제본.
    (원본은 실행 중인 GPU 큐가 import하는 파일이라 건드리지 않는다.)"""
    ps = PolySpec(spec.sigmoid_coeffs, 2.0, spec.exp_coeffs, spec.exp_interval, spec.recip_iterations, spec.steepness)
    params = {k: v.copy() for k, v in params0.items()}
    vel = {k: np.zeros_like(v) for k, v in params.items()}
    lrs = {"threshold": opt.node_lr(X.shape[1]), "alpha": opt.node_lr(X.shape[1]), "leaf_logits": opt.leaf_lr(Y.shape[1])}
    bounds = {"threshold": (opt.threshold_bound, 1 / opt.threshold_radius),
              "alpha": (opt.softmax_bound, 1 / opt.softmax_radius), "leaf_logits": (opt.softmax_bound, 1 / opt.softmax_radius)}
    hist = []
    for it in range(iterations):
        grads, _ = compute_gradients(X, Y, params, depth, spec.steepness, ps, balance_lambda=opt.balance,
                                     loss="mse" if sample_weight is None else "wmse", sample_weight=sample_weight)
        for k in params:
            vel[k] = grads[k] if it == 0 else opt.beta * vel[k] + grads[k]
            params[k] = params[k] - lrs[k] * vel[k]
            if opt.clamp:
                params[k] = H.clamp_plain(params[k], *bounds[k])
        hist.append({k: v.copy() for k, v in params.items()})
    return hist


def metrics(y: np.ndarray, prob: np.ndarray, K: int, prior: np.ndarray) -> dict:
    present = [c for c in range(K) if (y == c).any()]
    out = {}
    for tag, pred in (("", prob.argmax(1)), ("pc_", (prob / np.maximum(prior, 1e-12)).argmax(1))):
        out[tag + "acc"] = float((pred == y).mean())
        out[tag + "bal"] = float(np.mean([(pred[y == c] == c).mean() for c in present]))
        out[tag + "f1"] = float(f1_score(y, pred, labels=present, average="macro", zero_division=0))
        out[tag + "mcc"] = float(matthews_corrcoef(y, pred))
        if tag == "":
            out["cm"] = confusion_matrix(y, pred, labels=list(range(K))).tolist()
            if K == 2:
                minority = int(np.argmin(prior))
                out["f1_minority"] = float(f1_score(y, pred, pos_label=minority, average="binary", zero_division=0))
    p = np.clip(prob, 1e-9, None)
    p = p / p.sum(1, keepdims=True)
    try:
        if K == 2:
            minority = int(np.argmin(prior))
            out["auc"] = float(roc_auc_score(y == minority, p[:, minority]))
        else:
            out["auc"] = float(roc_auc_score(y, p[:, present] / p[:, present].sum(1, keepdims=True),
                                             labels=present, multi_class="ovr", average="macro"))
    except ValueError:
        out["auc"] = None
    return out


def job(args):
    ds, fold, name, seed = args
    Xtr_raw, Xte_raw, ytr, yte = evaluation_splits(ds)[fold]
    sc = fit_scaler(Xtr_raw)
    X, Xte = scale_features(sc, Xtr_raw), scale_features(sc, Xte_raw)
    K = int(max(ytr.max(), yte.max()) + 1)
    depth = 4 if K > 8 else 3
    prior = np.bincount(ytr, minlength=K) / len(ytr)
    R = free_models(len(ytr), X.shape[1], K, depth)
    w = class_weights(ytr, K, {**CONFIGS, **EXTRA_CONFIGS}[name])
    hists, ood_at = [], None
    with np.errstate(all="ignore"):
        for r in range(R):
            init = init_params(X.shape[1], K, depth, 1000 * r + 100003 * seed)
            h = emulate_w(X, one_hot_encode(ytr, K), depth, T.default_spec(), FULL, max(CHECKPOINTS), init, w)
            for i in range(9, len(h), 10):  # 10 iteration마다 구간 확인, 처음 이탈한 iteration 기록
                d = domain_report(X, {**h[i], "steepness": 8.0, "depth": depth})
                bad = (d["n_samples_gate_out_of_range"] > 0 or d["softmax_out_of_range"]
                       or not all(np.isfinite(v).all() for v in h[i].values()))
                if bad:
                    ood_at = i + 1 if ood_at is None else min(ood_at, i + 1)
                    break
            hists.append(h)
        rows = []
        for it in CHECKPOINTS:
            probs = [predict_proba(Xte, {**h[it - 1], "steepness": 8.0, "depth": depth}, DEFAULT_POLY_SPEC) for h in hists]
            ok = [p for p in probs if np.isfinite(p).all()]
            m = metrics(yte, np.mean(ok, 0), K, prior) if ok else None
            rows.append({"dataset": ds, "fold": fold, "config": name, "seed": seed, "R": R, "K": K, "depth": depth,
                         "iterations": it, "n_test": len(yte), "w_max": float(w.max()) if w is not None else 1.0,
                         "ood_first_it": ood_at, "metrics": m})
    return rows


def main():
    out = Path(sys.argv[1])
    n_proc = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    out.mkdir(parents=True, exist_ok=True)
    path = out / "runs.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["dataset"], r["fold"], r["config"], r["seed"]))
    jobs = [(ds, f, name, s) for ds in DATASETS for f in range(len(evaluation_splits(ds)))
            for name in CONFIGS for s in SEEDS if (ds, f, name, s) not in done]
    jobs += [(ds, f, name, s) for ds in EXTRA_DATASETS for f in range(len(evaluation_splits(ds)))
             for name in EXTRA_CONFIGS for s in SEEDS if (ds, f, name, s) not in done]
    jobs.sort(key=lambda j: j[0] in ("sepsis", "cover10k", "digits"))  # 큰 데이터셋을 뒤로(작은 것부터 결과 확인)
    print(f"{len(jobs)} jobs ({len(done)} done)", flush=True)
    with Pool(n_proc) as pool, open(path, "a") as f:
        for rows in pool.imap_unordered(job, jobs):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            last = rows[-1]
            print(json.dumps({k: last[k] for k in ("dataset", "fold", "config", "seed", "ood_first_it")}
                             | {"acc": last["metrics"] and round(last["metrics"]["acc"], 3),
                                "bal": last["metrics"] and round(last["metrics"]["bal"], 3)}), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
