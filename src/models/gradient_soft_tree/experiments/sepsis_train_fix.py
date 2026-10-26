"""sepsis 3단계(2026-10-26): 학습 쪽 보정 비교. CKKS 다항식 에뮬레이션, fold0, R=1.

cd src && PYTHONPATH=. python models/gradient_soft_tree/experiments/sepsis_train_fix.py  (결과: results/sepsis_diag/train_fix.jsonl)

levelpacked.tree.emulate와 같은 규칙(momentum + 노드/leaf lr + clamp)인데 compute_gradients에
loss/sample_weight를 넘길 수 있게 복제했다(원본은 mse 고정).
  mse_l1_*   : 현재 본실험 설정(balance λ=1, routing balance - 클래스 보정 아님)
  mse_l0_100 : λ=0
  wmse_*     : 클래스 빈도 역수 가중(평균 1), 소수 클래스 w≈6.8
  wmse_sqrt  : 역수의 제곱근 가중(중간 강도)
"""
from __future__ import annotations

import json
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import roc_auc_score

from core.data.dataset import evaluation_splits, fit_scaler, one_hot_encode, scale_features
from models.gradient_soft_tree.baseline.reference import (DEFAULT_POLY_SPEC, PolySpec, compute_gradients,
                                                          domain_report, init_params, predict_proba)
from models.gradient_soft_tree.levelpacked import tree as T
from models.gradient_soft_tree.experiments.ablation_study import FULL

CONFIGS = {  # 이름: (balance λ, 가중 방식, iterations)
    "mse_l1_100": (1.0, None, 100),
    "mse_l1_300": (1.0, None, 300),
    "mse_l0_100": (0.0, None, 100),
    "wmse_100": (1.0, "inv", 100),
    "wmse_300": (1.0, "inv", 300),
    "wmse_sqrt_100": (1.0, "sqrt", 100),
    "wmse_sqrt_300": (1.0, "sqrt", 300),
}


def bal_acc(y, pred):
    return float(np.mean([(pred[y == c] == c).mean() for c in (0, 1)]))


def emulate_w(X, Y, depth, spec, opt, iterations, params0, sample_weight):
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
            params[k] = T.H.clamp_plain(params[k] - lrs[k] * vel[k], *bounds[k])
        hist.append({k: v.copy() for k, v in params.items()})
    return hist


def job(name):
    lam, wkind, iters = CONFIGS[name]
    Xtr_raw, Xte_raw, ytr, yte = evaluation_splits("sepsis")[0]
    sc = fit_scaler(Xtr_raw)
    X, Xte = scale_features(sc, Xtr_raw), scale_features(sc, Xte_raw)
    K, depth = 2, 3
    prior = np.bincount(ytr, minlength=K) / len(ytr)
    w = None
    if wkind:
        cw = 1 / prior if wkind == "inv" else 1 / np.sqrt(prior)
        w = cw[ytr] / cw[ytr].mean()
    opt = replace(FULL, balance=lam)
    with np.errstate(all="ignore"):
        hist = emulate_w(X, one_hot_encode(ytr, K), depth, T.default_spec(), opt, iters, init_params(X.shape[1], K, depth, 0), w)
        ood = False
        for p in hist[9::10]:
            d = domain_report(X, {**p, "steepness": 8.0, "depth": depth})
            ood |= bool(d["n_samples_gate_out_of_range"] > 0 or d["softmax_out_of_range"])
        p = {**hist[-1], "steepness": 8.0, "depth": depth}
        p_te = predict_proba(Xte, p, DEFAULT_POLY_SPEC)
    am, pc = p_te.argmax(1), (p_te / prior).argmax(1)
    row = {"config": name, "auc": round(float(roc_auc_score(yte == 0, p_te[:, 0])), 4),
           "argmax_acc_bal": [round(float((am == yte).mean()), 3), round(bal_acc(yte, am), 3)],
           "prior_corr_acc_bal": [round(float((pc == yte).mean()), 3), round(bal_acc(yte, pc), 3)],
           "p0_max": round(float(p_te[:, 0].max()), 3), "out_of_domain": ood}
    print(json.dumps(row), flush=True)
    return row


if __name__ == "__main__":
    with Pool(len(CONFIGS)) as pool:
        pool.map(job, list(CONFIGS))
