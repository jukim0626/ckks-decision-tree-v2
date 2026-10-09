"""논문 실험 전 CPU 점검(PAPER_ROADMAP.md Phase 1~2): 전체 후보 데이터셋 x 평가 분할(3-fold CV / holdout)에서
CKKS와 같은 다항식으로 학습하는 plaintext 에뮬레이션을 돌려 정확도와 iteration 예산을 정한다. GPU 불필요.

방법
- safe: 제안 방식(momentum 0.9, lr 1, kappa 0.25, 합성 다항식 clamp) - 모든 데이터셋 공통 설정
- gd:   기존 vanilla GD, lr은 fold train 안의 validation으로 {1, 2, 4, 6} 중 선택
- sklearn DT(같은 depth), LogReg: plaintext 기준선
depth = 3, class 수가 8 초과면 4 (2026-10-08 결정).

iteration 예산 선택(test 누수 방지): 각 fold train을 다시 fit/val(75/25)로 나눠 safe 방식의 val 정확도 곡선을 보고,
모든 데이터셋 평균이 최고값의 0.5%p 안에 처음 들어오는 iteration을 공통 예산으로 고른다. 최종 test 정확도는 fold train
전체로 다시 학습해서 잰다.

python -m models.gradient_soft_tree.experiments.paper_screen <out.jsonl>
"""

from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import evaluation_splits, fit_scaler, one_hot_encode, scale_features  # noqa: E402
from models.gradient_soft_tree.baseline.reference import DEFAULT_POLY_SPEC, domain_report, init_params, predict  # noqa: E402
from models.gradient_soft_tree.levelpacked import tree as T  # noqa: E402

DATASETS = ["iris", "wine", "breast_cancer", "breast_cancer_nominal", "soybean15", "diabetes", "banknote",
            "digits", "cover10k", "sepsis"]
MAX_IT = 200
CHECKPOINTS = [10, 20, 30, 50, 100, 150, 200]
SAFE = T.OptConfig("mom", 1.0, 0.9, 0.25, True, leaf_k_scale=True)  # 10-08: leaf lr x K 포함
GD_LRS = [1.0, 2.0, 4.0, 6.0]


def depth_for(n_classes: int) -> int:
    return 3 if n_classes <= 8 else 4


def _curve(X, y, Xe, ye, K, depth, opt, seed=0):
    """iteration별 정확도(체크포인트)와 구간 이탈 여부."""
    hist = T.emulate(X, one_hot_encode(y, K), depth, T.default_spec(), opt, MAX_IT, init_params(X.shape[1], K, depth, seed))
    accs, safe = {}, True
    for it in CHECKPOINTS:
        p = {**hist[it - 1], "steepness": 8.0, "depth": depth}
        with np.errstate(all="ignore"):
            accs[it] = float((predict(Xe, p, DEFAULT_POLY_SPEC) == ye).mean())
    for h in hist:
        d = domain_report(X, {**h, "steepness": 8.0, "depth": depth})
        if d["n_samples_gate_out_of_range"] or d["softmax_out_of_range"] or not np.isfinite(d["max_abs_gate_input"]):
            safe = False
            break
    return accs, safe


def job(args):
    dataset, fold = args
    X_tr_raw, X_te_raw, y_tr, y_te = evaluation_splits(dataset)[fold]
    K = int(max(y_tr.max(), y_te.max()) + 1)
    depth = depth_for(K)
    sc = fit_scaler(X_tr_raw)
    X, Xte = scale_features(sc, X_tr_raw), scale_features(sc, X_te_raw)
    # fold train 안의 validation (iteration 예산, GD lr 선택용)
    Xf_raw, Xv_raw, yf, yv = train_test_split(X_tr_raw, y_tr, test_size=0.25, random_state=0, stratify=y_tr)
    scv = fit_scaler(Xf_raw)
    Xf, Xv = scale_features(scv, Xf_raw), scale_features(scv, Xv_raw)

    out = {"dataset": dataset, "fold": fold, "depth": depth, "n_train": len(y_tr), "n_test": len(y_te), "K": K,
           "majority": float(np.bincount(y_te).max() / len(y_te))}
    out["safe_val"], _ = _curve(Xf, yf, Xv, yv, K, depth, SAFE)
    out["safe_test"], out["safe_domain_ok"] = _curve(X, y_tr, Xte, y_te, K, depth, SAFE)
    best = None
    for lr in GD_LRS:
        v, _ = _curve(Xf, yf, Xv, yv, K, depth, T.OptConfig("gd", lr))
        if best is None or v[30] > best[0]:
            best = (v[30], lr)
    out["gd_lr"] = best[1]
    out["gd_test"], out["gd_domain_ok"] = _curve(X, y_tr, Xte, y_te, K, depth, T.OptConfig("gd", best[1]))
    out["sk_tree"] = float(DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X, y_tr).score(Xte, y_te))
    out["logreg"] = float(LogisticRegression(max_iter=3000).fit(X, y_tr).score(Xte, y_te))
    return out


def g(r, key, it):
    d = r[key]
    return d[str(it)] if str(it) in d else d[it]


def summarize(rows: list[dict]) -> None:
    # 공통 iteration 예산: safe val 곡선의 데이터셋 평균
    mean_val = {it: np.mean([np.mean([r["safe_val"][str(it)] if str(it) in r["safe_val"] else r["safe_val"][it]
                                      for r in rows if r["dataset"] == ds]) for ds in DATASETS]) for it in CHECKPOINTS}
    top = max(mean_val.values())
    budget = next(it for it in CHECKPOINTS if mean_val[it] >= top - 0.005)
    print("safe val 정확도(데이터셋 평균) by iteration:", {it: round(v, 3) for it, v in mean_val.items()})
    print(f"-> 공통 iteration 예산 = {budget}")
    for label, sel in [("class<=3", lambda r: r["K"] <= 3), ("class>3", lambda r: r["K"] > 3)]:
        dss = sorted({r["dataset"] for r in rows if sel(r)})
        mv = {it: np.mean([np.mean([g(r, "safe_val", it) for r in rows if r["dataset"] == ds]) for ds in dss]) for it in CHECKPOINTS}
        b = next(it for it in CHECKPOINTS if mv[it] >= max(mv.values()) - 0.005)
        print(f"   {label} ({', '.join(dss)}): 예산 {b} | " + " ".join(f"@{it}:{v:.3f}" for it, v in mv.items()))
    print()

    print(f"{'dataset':22s} {'d':>2s} {'folds':>5s} {'다수':>6s} {'sk트리':>7s} {'LogReg':>7s} "
          f"{'GD@'+str(budget):>9s} {'safe@30':>8s} {'safe@50':>8s} {'safe@'+str(budget):>13s} {'safe@200':>8s}  구간")
    for ds in DATASETS:
        rs = [r for r in rows if r["dataset"] == ds]
        if not rs:
            continue
        f = lambda xs: f"{np.mean(xs):.3f}±{np.std(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"
        print(f"{ds:22s} {rs[0]['depth']:>2d} {len(rs):>5d} {np.mean([r['majority'] for r in rs]):>6.3f} "
              f"{np.mean([r['sk_tree'] for r in rs]):>7.3f} {np.mean([r['logreg'] for r in rs]):>7.3f} "
              f"{np.mean([g(r, 'gd_test', budget) for r in rs]):>9.3f} {np.mean([g(r, 'safe_test', 30) for r in rs]):>8.3f} "
              f"{np.mean([g(r, 'safe_test', 50) for r in rs]):>8.3f} "
              f"{f([g(r, 'safe_test', budget) for r in rs]):>13s} {np.mean([g(r, 'safe_test', 200) for r in rs]):>8.3f}  "
              f"safe={'OK' if all(r['safe_domain_ok'] for r in rs) else '이탈'} gd={'OK' if all(r['gd_domain_ok'] for r in rs) else '이탈'}")


def main() -> None:
    out = Path(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "summary":
        summarize([json.loads(l) for l in open(out)])
        return
    jobs = [(ds, f) for ds in DATASETS for f in range(len(evaluation_splits(ds)))]
    with Pool(min(len(jobs), 24)) as pool, open(out, "w") as fh:
        rows = []
        for r in pool.imap_unordered(job, jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
            print(f"[done] {r['dataset']} fold {r['fold']}", flush=True)
    summarize(rows)


if __name__ == "__main__":
    main()
