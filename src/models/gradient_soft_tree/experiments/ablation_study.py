"""논문 ablation(2026-10-09). CKKS 다항식 에뮬레이션(plaintext, CPU) - GPU 본 실험과 같은 분할/예산/초기값.

구성요소를 하나씩 빼서 본 방법(full)과 비교한다:
  full        momentum + 노드 lr 0.25F + leaf lr K + clamp + balance λ=1 + multi-start(R=free_models)
  no_balance  λ=0
  no_multi    R=1
  no_both     λ=0, R=1 (2026-10-08 GPU 1차 큐와 같은 설정)
  no_leafK    leaf lr = η (K 보정 없음)
  no_clamp    clamp 없음(다항식 구간 보장 없음)
  true_fn          full과 같은 규칙, sigmoid/softmax만 진짜 함수(학습+추론) - "다항식 근사 손실" 열(2026-10-20)
  true_fn_noclamp  true_fn에서 clamp도 뺌(제약 없는 평문 soft tree)
예산: 클래스 <= 3은 100, 다중 클래스는 300 iteration. 평가 지표: accuracy, balanced accuracy, 학습 중 다항식 구간 이탈 여부.

python -m models.gradient_soft_tree.experiments.ablation_study <out_dir> [config1,config2,...]
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import evaluation_splits, fit_scaler, one_hot_encode, scale_features  # noqa: E402
from models.gradient_soft_tree.baseline.reference import DEFAULT_POLY_SPEC, domain_report, init_params, predict_proba  # noqa: E402
from models.gradient_soft_tree.levelpacked import tree as T  # noqa: E402
from models.gradient_soft_tree.levelpacked.layout import free_models  # noqa: E402

DATASETS = ["iris", "wine", "breast_cancer", "breast_cancer_nominal", "diabetes", "banknote",
            "soybean15", "digits", "cover10k", "sepsis"]
FULL = T.OptConfig("mom", 1.0, 0.9, 0.25, True, leaf_k_scale=True, balance=1.0)
CONFIGS = {  # 이름: (OptConfig, multi-start 사용 여부)
    "full": (FULL, True),
    "no_balance": (replace(FULL, balance=0.0), True),
    "no_multi": (FULL, False),
    "no_both": (replace(FULL, balance=0.0), False),
    "no_leafK": (replace(FULL, leaf_k_scale=False), True),
    "no_clamp": (replace(FULL, clamp=False), True),
    "true_fn": (FULL, True),
    "true_fn_noclamp": (replace(FULL, clamp=False), True),
}


def balanced_acc(y, p, K):
    return float(np.mean([(p[y == c] == c).mean() for c in range(K) if (y == c).any()]))


def job(args):
    ds, fold, name = args
    opt, multi = CONFIGS[name]
    Xtr_raw, Xte_raw, ytr, yte = evaluation_splits(ds)[fold]
    sc = fit_scaler(Xtr_raw)
    X, Xte = scale_features(sc, Xtr_raw), scale_features(sc, Xte_raw)
    K = int(max(ytr.max(), yte.max()) + 1)
    depth = 4 if K > 8 else 3
    T_budget = 100 if K <= 3 else 300
    Y = one_hot_encode(ytr, K)
    R = free_models(len(ytr), X.shape[1], K, depth) if multi else 1
    true_fn = name.startswith("true_fn")
    finals, out_of_domain = [], False
    with np.errstate(all="ignore"):
        for r in range(R):
            hist = T.emulate(X, Y, depth, T.default_spec(), opt, T_budget, init_params(X.shape[1], K, depth, 1000 * r),
                              true_fn=true_fn)
            for p in hist[9::10]:  # 10 iteration마다 구간 확인
                d = domain_report(X, {**p, "steepness": 8.0, "depth": depth})
                out_of_domain |= bool(d["n_samples_gate_out_of_range"] > 0 or d["softmax_out_of_range"]
                                      or not all(np.isfinite(v).all() for v in p.values()))
            finals.append({**hist[-1], "steepness": 8.0, "depth": depth})
        probs = [predict_proba(Xte, p, "true" if true_fn else DEFAULT_POLY_SPEC) for p in finals]
    ok = [p for p in probs if np.isfinite(p).all()]
    pred = np.mean(ok, 0).argmax(1) if ok else np.zeros_like(yte)
    row = {"dataset": ds, "fold": fold, "config": name, "R": R, "depth": depth, "iterations": T_budget,
           "acc": float((pred == yte).mean()), "bal": balanced_acc(yte, pred, K), "out_of_domain": out_of_domain}
    print(json.dumps(row), flush=True)
    return row


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(CONFIGS)
    jobs = [(ds, f, c) for ds in DATASETS for f in range(len(evaluation_splits(ds))) for c in names]
    with Pool(int(os.environ.get("N_WORKERS", "12"))) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    with open(out / ("ablation.jsonl" if len(sys.argv) <= 2 else f"ablation_{'_'.join(names)}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
