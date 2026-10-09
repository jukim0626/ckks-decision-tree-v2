"""local minimum 스트레스 테스트(2026-10-09, PAPER_ROADMAP Phase 1). CKKS 다항식 에뮬레이션(plaintext, CPU).

목적: 논문에서 "실패(나쁜 국소해 / routing 쏠림 / 정체)가 실제로 생기는 조건"을 보이고, 대책이 그 실패를 줄이는지,
그리고 이미 정체된 상태에서 빠져나올 수 있는지를 숫자로 보인다. 실제 데이터셋에서는 실패가 드물었으므로 일부러 어려운
조건을 만든다.

조건(18개) = 합성 데이터 6종 x 초기값 3종
  데이터(train 400 / test 400, x in [-1,1]): xor(d2), xor_noise(무관 feature 8개 추가, d3), parity3(d3), checker(4x4, d4),
                                          circles(d3), spirals(d4)
  초기값: default(N(0,0.1)), big(N(0,1.0)), collapsed(threshold 전부 +0.9 -> 모든 샘플이 한쪽으로, alpha 0)
방법(seed 64개, 같은 seed 짝비교): base(대책 없음) / balance(λ=1) / noise(σ=0.01, 외부 pilot 후보) / continuation
  + multi-start R=4,8(base·balance seed 묶음): 앙상블 평균 / training loss 최선 선택. 이론 실패율 p^R과 비교
L* = 각 조건의 모든 실행(+ seed 16개 x 2000it) 중 최저 training loss. 정규화 gap = (L-L*)/(L0-L*), 실패 = gap >= 0.3
checkpoint 고정 탈출 실험: base에서 실패한 run(조건당 최대 8개)의 상태(파라미터+velocity)에서 처치별로 이어서 학습
  처치: continue(그대로 더) / kick0.03 / kick0.1(한 번 노이즈) / balance(λ=1 켜기) / reset_v(velocity 0)
  처치마다 무작위 시도 8번 -> "출발점보다 낮은 loss 도달 비율", "최선 근처(gap<0.05) 도달 비율"

python -m models.gradient_soft_tree.experiments.stress_test <out_dir>
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
from models.gradient_soft_tree.levelpacked.he_ops import clamp_plain  # noqa: E402

SPEC = R.DEFAULT_POLY_SPEC
BOUNDS = {"threshold": (0.95, 1 / 8), "alpha": (2.45, 1 / 14), "leaf_logits": (2.45, 1 / 14)}
T = 150
N_SEEDS = 64
FAIL = 0.30
NEAR = 0.05
DATA_DEPTH = {"xor": 2, "xor_noise": 3, "parity3": 3, "checker": 4, "circles": 3, "spirals": 4}
INITS = ["default", "big", "collapsed"]
METHODS = {"base": {}, "balance": {"balance": 1.0}, "noise": {"noise": 0.01}, "continuation": {"cont": True}}


def make_data(name: str, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    if name in ("xor", "xor_noise"):
        X = rng.uniform(-1, 1, (n, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        if name == "xor_noise":
            X = np.hstack([X, rng.uniform(-1, 1, (n, 8))])
    elif name == "parity3":
        X = rng.uniform(-1, 1, (n, 3))
        y = ((X > 0).sum(1) % 2).astype(int)
    elif name == "checker":
        X = rng.uniform(-1, 1, (n, 2))
        y = ((np.floor((X[:, 0] + 1) * 2) + np.floor((X[:, 1] + 1) * 2)) % 2).astype(int)
    elif name == "circles":
        X = rng.uniform(-1, 1, (n, 2))
        y = (np.hypot(X[:, 0], X[:, 1]) < 0.6).astype(int)
    else:  # spirals
        t = rng.uniform(0.25, 1.0, n) * 3 * np.pi
        lab = rng.integers(0, 2, n)
        r = t / (3 * np.pi)
        ang = t + np.pi * lab
        X = np.column_stack([r * np.cos(ang), r * np.sin(ang)]) + rng.normal(0, 0.03, (n, 2))
        X = np.clip(X, -1, 1)
        y = lab
    return X, y


_D: dict = {}


def data(name: str):
    if name not in _D:
        rng = np.random.default_rng({"xor": 1, "xor_noise": 2, "parity3": 3, "checker": 4, "circles": 5, "spirals": 6}[name])
        X, y = make_data(name, 800, rng)
        _D[name] = (X[:400], y[:400], X[400:], y[400:])
    return _D[name]


def init(name: str, F: int, K: int, depth: int, seed: int, mode: str) -> dict:
    P = R.init_params(F, K, depth, seed)
    if mode == "big":
        rng = np.random.default_rng(seed + 777)
        P = {k: np.clip(rng.normal(0, 1.0, v.shape), -BOUNDS[k][0], BOUNDS[k][0]) for k, v in P.items()}
    elif mode == "collapsed":
        P["threshold"][:] = 0.9
        P["alpha"][:] = 0.0
    return P


def run(X, y, depth, P, V, t0, T_run, balance=0.0, noise=0.0, cont=False, seed=0, kick=0.0):
    """현재 방법의 update(momentum 0.9, lr 1, 노드 lr 0.25F, leaf lr K, 합성 clamp, MSE)를 T_run iteration."""
    K = int(y.max() + 1)
    F = X.shape[1]
    Y = one_hot_encode(y, K)
    lrs = {"alpha": max(1.0, 0.25 * F), "threshold": max(1.0, 0.25 * F), "leaf_logits": float(K)}
    rng = np.random.default_rng(50_000 + seed)
    P = {k: v.copy() for k, v in P.items()}
    V = {k: v.copy() for k, v in V.items()}
    if kick:
        P = {k: clamp_plain(v + kick * BOUNDS[k][0] * rng.standard_normal(v.shape), *BOUNDS[k]) for k, v in P.items()}
    for t in range(t0 + 1, t0 + T_run + 1):
        c = min(1.0, 0.25 + 0.75 * t / (0.6 * T)) if cont else 1.0
        with np.errstate(all="ignore"):
            g, _ = R.compute_gradients(X, Y, P, depth, approx=SPEC, gate_scale=c, balance_lambda=balance)
        for k in P:
            V[k] = 0.9 * V[k] + g[k] if t > 1 else g[k]
            new = P[k] - lrs[k] * V[k]
            if noise:
                new = new + noise * BOUNDS[k][0] * rng.standard_normal(new.shape)
            P[k] = clamp_plain(new, *BOUNDS[k])
    with np.errstate(all="ignore"):
        L = float(R.compute_gradients(X, Y, P, depth, approx=SPEC)[1])
    return P, V, L


def proba(X, P, depth):
    with np.errstate(all="ignore"):
        return R.predict_proba(X, {**P, "steepness": 8.0, "depth": depth}, SPEC)


def job_main(args):
    name, mode, method, seed, T_run = args
    X, y, Xte, yte = data(name)
    depth = DATA_DEPTH[name]
    P0 = init(name, X.shape[1], 2, depth, seed, mode)
    V0 = {k: np.zeros_like(v) for k, v in P0.items()}
    with np.errstate(all="ignore"):
        L0 = float(R.compute_gradients(X, one_hot_encode(y, 2), P0, depth, approx=SPEC)[1])
    P, V, L = run(X, y, depth, P0, V0, 0, T_run, seed=seed, **METHODS.get(method, {}))
    out = {"data": name, "init": mode, "method": method, "seed": seed, "T": T_run, "L0": L0, "L": L,
           "test_proba": proba(Xte, P, depth).tolist(), "train_acc": float((proba(X, P, depth).argmax(1) == y).mean())}
    if method == "base" and T_run == T:
        out["state"] = {"P": {k: v.tolist() for k, v in P.items()}, "V": {k: v.tolist() for k, v in V.items()}}
    return out


def job_escape(args):
    name, mode, seed, treat, trial, state = args
    X, y, _, _ = data(name)
    depth = DATA_DEPTH[name]
    P = {k: np.array(v) for k, v in state["P"].items()}
    V = {k: np.array(v) for k, v in state["V"].items()}
    kw = {"continue": {}, "kick0.03": {"kick": 0.03}, "kick0.1": {"kick": 0.1}, "balance": {"balance": 1.0}}.get(treat, {})
    if treat == "reset_v":
        V = {k: np.zeros_like(v) for k, v in V.items()}
    _, _, L = run(X, y, depth, P, V, T, T, seed=seed * 100 + trial, **kw)
    return {"data": name, "init": mode, "seed": seed, "treat": treat, "trial": trial, "L": L}


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    workers = int(os.environ.get("N_WORKERS", "12"))
    conds = [(d, m) for d in DATA_DEPTH for m in INITS]
    jobs = [(d, m, meth, s, T) for d, m in conds for meth in METHODS for s in range(N_SEEDS)]
    jobs += [(d, m, "long", s, 2000) for d, m in conds for s in range(16)]
    print(f"[stress] main {len(jobs)} runs", flush=True)
    with Pool(workers) as pool:
        rows = pool.map(job_main, jobs, chunksize=4)
    with open(out / "main.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "state"}) + "\n")

    Ls = {c: min(r["L"] for r in rows if (r["data"], r["init"]) == c) for c in conds}
    gap = lambda r: (r["L"] - Ls[(r["data"], r["init"])]) / max(r["L0"] - Ls[(r["data"], r["init"])], 1e-12)
    esc_jobs = []
    for c in conds:
        failed = [r for r in rows if (r["data"], r["init"]) == c and r["method"] == "base" and gap(r) >= FAIL][:8]
        for r in failed:
            for treat in ["continue", "kick0.03", "kick0.1", "balance", "reset_v"]:
                for trial in range(8 if treat.startswith("kick") else 1):
                    esc_jobs.append((c[0], c[1], r["seed"], treat, trial, r["state"]))
    print(f"[stress] escape {len(esc_jobs)} runs", flush=True)
    with Pool(workers) as pool:
        esc = pool.map(job_escape, esc_jobs, chunksize=2)
    with open(out / "escape.jsonl", "w") as f:
        for r in esc:
            f.write(json.dumps(r) + "\n")
    (out / "lstar.json").write_text(json.dumps({f"{a}|{b}": v for (a, b), v in Ls.items()}))
    print("[stress] done", flush=True)


if __name__ == "__main__":
    main()
