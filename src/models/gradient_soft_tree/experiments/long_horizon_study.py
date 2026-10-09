"""장시간 학습에서 loss가 다시 커지는 원인 분석(2026-10-09). cancer는 2000 iteration에서 loss 0.094 -> 0.35~0.40.

질문: 모든 파라미터를 동시에 업데이트해서인가, 노드 lr이 커서인가(cancer F=30 -> 7.5), momentum + clamp(projection) 상호작용인가.
변형(모두 CKKS 다항식 에뮬레이션, (g)만 진짜 sigmoid):
  a base        : 지금 방식(모든 파라미터 동시, momentum 0.9, 노드 lr = max(1, .25F), leaf lr = K, clamp)
  b alternate   : 홀수 iteration은 leaf만, 짝수는 노드(threshold, alpha)만
  c level_cycle : iteration마다 한 level의 노드만(+ leaf는 매번) 돌아가며
  d kappa0.1    : 노드 lr 축소
  e decay_sqrt  : lr * 1/sqrt(t/100+1) 감쇠 / e2 cosine 감쇠(T에서 0.05배)
  f beta0.5     : momentum 약화
  g no_clamp    : clamp 없이 진짜 sigmoid(근사 구간 무시) - clamp가 원인인지 분리
  h gd          : momentum 없는 GD(같은 lr) + clamp
기록: loss 곡선(50 간격), 파라미터별 이동량, clamp 경계 비율, gradient norm.

python -m models.gradient_soft_tree.experiments.long_horizon_study <out.jsonl> [datasets...]
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import one_hot_encode  # noqa: E402
from models.gradient_soft_tree.baseline import reference as R  # noqa: E402
from models.gradient_soft_tree.experiments.global_opt_study import BOUNDS, _data  # noqa: E402
from models.gradient_soft_tree.levelpacked.he_ops import clamp_plain  # noqa: E402

T_LONG = 2000
VARIANTS = ["a_base", "b_alternate", "c_level_cycle", "d_kappa0.1", "e_decay_sqrt", "e2_cosine", "f_beta0.5", "g_no_clamp", "h_gd"]


def run(args):
    ds, variant, seed = args
    X, y, Xte, yte, K, depth = _data(ds, 0)
    F = X.shape[1]
    Y = one_hot_encode(y, K)
    approx = "true" if variant == "g_no_clamp" else R.DEFAULT_POLY_SPEC
    P = R.init_params(F, K, depth, seed)
    V = {k: np.zeros_like(v) for k, v in P.items()}
    kappa = 0.1 if variant == "d_kappa0.1" else 0.25
    beta = 0.5 if variant == "f_beta0.5" else (0.0 if variant == "h_gd" else 0.9)
    base_lr = {"alpha": max(1.0, kappa * F), "threshold": max(1.0, kappa * F), "leaf_logits": float(K)}
    n_internal = (1 << depth) - 1
    level_of = np.concatenate([[l] * (1 << l) for l in range(depth)])
    curve, moves, clampr, gnorm = {}, {k: {} for k in P}, {}, {}
    prev = {k: v.copy() for k, v in P.items()}
    for t in range(1, T_LONG + 1):
        with np.errstate(all="ignore"):
            g, L = R.compute_gradients(X, Y, P, depth, approx=approx)
        mult = 1.0
        if variant == "e_decay_sqrt":
            mult = 1 / np.sqrt(t / 100 + 1)
        elif variant == "e2_cosine":
            mult = 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * t / T_LONG))
        for k in P:
            mask = np.ones_like(P[k])
            if variant == "b_alternate":
                active = (k == "leaf_logits") if t % 2 else (k != "leaf_logits")
                mask *= float(active)
            if variant == "c_level_cycle" and k != "leaf_logits":
                mask = (level_of == (t % depth))[:, None] * np.ones_like(P[k])
            V[k] = beta * V[k] + g[k] * mask if t > 1 else g[k] * mask
            new = P[k] - base_lr[k] * mult * V[k]
            P[k] = new if variant == "g_no_clamp" else clamp_plain(new, *BOUNDS[k])
        if t % 50 == 0:
            curve[t] = float(L)
            for k in P:
                moves[k][t] = float(np.abs(P[k] - prev[k]).max())
                prev[k] = P[k].copy()
            clampr[t] = float(np.mean(np.concatenate([(np.abs(P[k]) >= BOUNDS[k][0] - 1e-3).ravel() for k in P])))
            gnorm[t] = {k: float(np.linalg.norm(g[k])) for k in g}
    with np.errstate(all="ignore"):
        acc = float((R.predict(Xte, {**P, "steepness": 8.0, "depth": depth}, approx) == yte).mean())
    return {"dataset": ds, "variant": variant, "seed": seed, "curve": curve, "moves": moves, "clamp_rate": clampr,
            "grad_norm": gnorm, "test_acc": acc}


def main():
    out = Path(sys.argv[1])
    dss = sys.argv[2:] or ["breast_cancer", "wine", "banknote"]
    jobs = [(ds, v, s) for ds in dss for v in VARIANTS for s in range(4)]
    with Pool(int(os.environ.get("N_WORKERS", "12"))) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    out.write_text("\n".join(json.dumps(r) for r in rows))
    for ds in dss:
        print(f"\n== {ds}: training loss(중앙값, seed 4개) at iteration ==")
        cps = [150, 300, 500, 1000, 1500, 2000]
        print(f"{'variant':15s} " + " ".join(f"{c:>7d}" for c in cps) + "   min(위치)   test acc   clamp비율@2000")
        for v in VARIANTS:
            rs = [r for r in rows if r["dataset"] == ds and r["variant"] == v]
            cur = np.array([[r["curve"][c] for c in sorted(r["curve"], key=int)] for r in rs])
            med = np.median(cur, 0)
            ks = sorted(rs[0]["curve"], key=int)
            mi = int(np.argmin(med))
            print(f"{v:15s} " + " ".join(f"{med[ks.index(c)]:7.4f}" for c in cps)
                  + f"   {med[mi]:.4f}(@{ks[mi]})   {np.mean([r['test_acc'] for r in rs]):.3f}"
                  + f"   {np.mean([r['clamp_rate'][2000] for r in rs]):.2f}")


if __name__ == "__main__":
    main()
