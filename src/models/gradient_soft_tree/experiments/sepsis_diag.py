"""sepsis 진단(2026-10-26): argmax 정확도 .926(=다수 클래스)일 때 모델이 실제로 순위(ranking)는 배웠는지 AUC로 확인.

cd src && PYTHONPATH=. python models/gradient_soft_tree/experiments/sepsis_diag.py  (결과: results/sepsis_diag/diag.jsonl)

ablation_study.job과 같은 분할/예산/초기값/multi-start로 CKKS 다항식 에뮬레이션(full)과 true_fn을 학습하고,
test 확률로 AUC, prior 보정, train에서 고른 threshold의 balanced acc를 잰다. 비교: LogReg, CART d3.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from core.data.dataset import evaluation_splits, fit_scaler, one_hot_encode, scale_features
from models.gradient_soft_tree.baseline.reference import DEFAULT_POLY_SPEC, predict_proba, init_params
from models.gradient_soft_tree.levelpacked import tree as T
from models.gradient_soft_tree.levelpacked.layout import free_models
from models.gradient_soft_tree.experiments.ablation_study import FULL


def bal_acc(y, pred):
    return float(np.mean([(pred[y == c] == c).mean() for c in (0, 1)]))


def report(name, p_tr, ytr, p_te, yte, prior):
    """p_*: 클래스 0(소수) 확률. argmax / prior 보정 / train에서 고른 threshold 세 가지 판정."""
    s_tr, s_te = p_tr[:, 0], p_te[:, 0]
    argmax = p_te.argmax(1)
    corr = (p_te / prior).argmax(1)  # p(c|x)/p(c) 비교 = 균형 prior 가정의 Bayes 판정
    cands = np.quantile(s_tr, np.linspace(0.01, 0.99, 197))
    t = max(cands, key=lambda t: bal_acc(ytr, np.where(s_tr >= t, 0, 1)))
    thr = np.where(s_te >= t, 0, 1)
    row = {"model": name, "auc": float(roc_auc_score(yte == 0, s_te)),
           "argmax": [float((argmax == yte).mean()), bal_acc(yte, argmax)],
           "prior_corr": [float((corr == yte).mean()), bal_acc(yte, corr)],
           "train_thr": [float((thr == yte).mean()), bal_acc(yte, thr)],
           "p0_test_quantiles": [round(float(q), 4) for q in np.quantile(s_te, [0, .1, .5, .9, 1])]}
    print(json.dumps(row), flush=True)
    return row


def main():
    Xtr_raw, Xte_raw, ytr, yte = evaluation_splits("sepsis")[0]
    sc = fit_scaler(Xtr_raw)
    X, Xte = scale_features(sc, Xtr_raw), scale_features(sc, Xte_raw)
    K, depth, iters = 2, 3, 100
    prior = np.bincount(ytr, minlength=K) / len(ytr)
    print(json.dumps({"n_train": len(ytr), "n_test": len(yte), "prior": prior.tolist()}), flush=True)

    for name, true_fn in (("emul_poly(full)", False), ("emul_true_fn", True)):
        R = free_models(len(ytr), X.shape[1], K, depth)
        finals = []
        with np.errstate(all="ignore"):
            for r in range(R):
                hist = T.emulate(X, one_hot_encode(ytr, K), depth, T.default_spec(), FULL, iters,
                                 init_params(X.shape[1], K, depth, 1000 * r), true_fn=true_fn)
                finals.append({**hist[-1], "steepness": 8.0, "depth": depth})
            spec = "true" if true_fn else DEFAULT_POLY_SPEC
            p_tr = np.mean([predict_proba(X, p, spec) for p in finals], 0)
            p_te = np.mean([predict_proba(Xte, p, spec) for p in finals], 0)
        report(f"{name} R={R}", p_tr, ytr, p_te, yte, prior)

    for name, clf in (("logreg", LogisticRegression(max_iter=1000)),
                      ("cart_d3", DecisionTreeClassifier(max_depth=3, random_state=0))):
        clf.fit(X, ytr)
        report(name, clf.predict_proba(X), ytr, clf.predict_proba(Xte), yte, prior)


if __name__ == "__main__":
    sys.exit(main())
