"""local minimum 문제 측정과 완화 방법 비교 (plaintext, CKKS와 같은 다항식 에뮬레이션, CPU).

질문: 정해진 iteration 예산 안에서 제안 방식이 도달하는 training loss가 "알려진 최선값"에서 얼마나 먼가, 그리고 그 gap을
HE에서 싸게 쓸 수 있는 방법으로 줄일 수 있는가. non-convex라 global minimum 보장은 불가능하므로 경험적으로 측정한다.

- 알려진 최선값 L*: seed 50개 x 2000 iteration(+ 이 스크립트의 모든 실행 중 최저값)
- 정규화 gap = (L - L*) / (L0 - L*), L0 = 초기 loss. 0.05 이하면 "최선값 근처"
- 방법(모두 같은 예산 T: class<=3이면 150, 아니면 200)
  base       : 제안 방식(momentum 0.9, lr 1, kappa 0.25, leaf lr x K, 합성 clamp)
  A multistart: 서로 다른 seed 8개를 평균(앙상블) / training loss 최선 선택(학습 후 client가 loss 8개만 복호화)
  B continue : gate 입력 배율 c(t)를 c0 -> 1로 선형 증가(실효 steepness 8*c, 같은 다항식, 구간 안전 유지)
  C noise    : clamp 직전 데이터 무관 perturbation sigma_t * 박스폭 * xi (xi ~ N(0,1), 공개 seed)
  D balance  : 나눗셈 없는 balance penalty (reference.compute_gradients의 balance_lambda)
- 하이퍼파라미터는 training loss gap(최적화 품질)으로 고른다 - test는 보고만 한다.

python -m models.gradient_soft_tree.experiments.global_opt_study <out_dir>
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import evaluation_splits, fit_scaler, one_hot_encode, scale_features  # noqa: E402
from models.gradient_soft_tree.baseline import reference as R  # noqa: E402
from models.gradient_soft_tree.levelpacked.he_ops import clamp_plain  # noqa: E402

SPEC = R.DEFAULT_POLY_SPEC
DATASETS = ["iris", "wine", "breast_cancer", "breast_cancer_nominal", "diabetes", "banknote", "soybean15", "digits"]
BOUNDS = {"threshold": (0.95, 1 / 8), "alpha": (2.45, 1 / 14), "leaf_logits": (2.45, 1 / 14)}
N_WORKERS = int(os.environ.get("N_WORKERS", "12"))


@dataclass(frozen=True)
class Method:
    name: str = "base"
    c0: float = 1.0          # continuation 시작 배율(1이면 없음)
    c_frac: float = 0.0      # 배율이 1에 도달하는 시점(예산 대비 비율)
    sigma0: float = 0.0      # perturbation 크기(박스폭 대비)
    sched: str = "lin"       # lin: T/2에서 0까지 선형 감쇠 / sqrt: 1/sqrt(t)
    balance: float = 0.0
    lr_sched: str = "const"  # const | linear(T에서 0) | cosine(T에서 0) | sqrt(1/sqrt(t/T_ref+1))  - 2026-10-09
    node_cap: float = 0.0    # 노드 lr 상한(0이면 없음): min(0.25F, cap)
    step_clip: float = 0.0   # 업데이트량 원소별 상한(박스폭 대비 비율, 0이면 없음) - 합성 clamp로 구현 가능
    loss: str = "mse"        # mse | wmse(클래스 빈도 역수 가중) | ece(expected CE) - 2026-10-09
    lr_mult: float = 1.0     # 전체 lr 배율(loss마다 gradient 크기가 달라 따로 맞춤)


_DATA: dict = {}


def _data(ds: str, fold: int = 0, inner: bool = False):
    """inner=True면 fold train을 fit/val(75/25)로 다시 나눠 (fit, val)을 (train, test) 자리에 돌려준다(하이퍼파라미터 선택용)."""
    key = (ds, fold, inner)
    if key not in _DATA:
        Xr, Xter, y, yte = evaluation_splits(ds)[fold]
        if inner:
            from sklearn.model_selection import train_test_split
            Xr, Xter, y, yte = train_test_split(Xr, y, test_size=0.25, random_state=0, stratify=y)
        sc = fit_scaler(Xr)
        K = int(max(y.max(), yte.max()) + 1)
        _DATA[key] = (scale_features(sc, Xr), y, scale_features(sc, Xter), yte, K, 3 if K <= 8 else 4)
    return _DATA[key]


def budget(K: int) -> int:
    return 150 if K <= 3 else 200


def yhat(X, P, depth, scale=1.0):
    n = X.shape[0]
    probs, i = [np.ones(n)], 0
    for _ in range(depth):
        nxt = []
        for pp in probs:
            w = R._softmax(P["alpha"][i], SPEC)
            g = R._gate_values(scale * (X - P["threshold"][i]), 8.0, SPEC) @ w
            nxt += [pp * (1 - g), pp * g]
            i += 1
        probs = nxt
    ld = [R._softmax(P["leaf_logits"][l], SPEC) for l in range(len(probs))]
    return sum(np.outer(probs[l], ld[l]) for l in range(len(probs))), probs


def routing(X, P, depth) -> dict:
    _, leaf = yhat(X, P, depth)
    mass = np.array([p.mean() for p in leaf])
    m = mass / mass.sum()
    ent = float(-(m * np.log(np.clip(m, 1e-12, None))).sum() / np.log(len(m)))
    # 노드 균형 |alpha_i - 1/2|, alpha_i = sum P^i g_i / sum P^i
    n = X.shape[0]
    probs, i, dev = [np.ones(n)], 0, []
    for _ in range(depth):
        nxt = []
        for pp in probs:
            w = R._softmax(P["alpha"][i], SPEC)
            g = R._gate_values(X - P["threshold"][i], 8.0, SPEC) @ w
            dev.append(abs((pp * g).sum() / max(pp.sum(), 1e-12) - 0.5))
            nxt += [pp * (1 - g), pp * g]
            i += 1
        probs = nxt
    return {"dead_leaves": int((mass < 0.01).sum()), "leaf_entropy": ent, "node_imbalance": float(np.mean(dev))}


def train(ds: str, seed: int, m: Method, T: int | None = None, fold: int = 0, inner: bool = False,
          depth: int | None = None) -> dict:
    X, y, Xte, yte, K, depth0 = _data(ds, fold, inner)
    depth = depth or depth0
    T = T or budget(K)
    F = X.shape[1]
    Y = one_hot_encode(y, K)
    P = R.init_params(F, K, depth, seed)
    V = {k: np.zeros_like(v) for k, v in P.items()}
    node_lr = max(1.0, 0.25 * F) if not m.node_cap else min(max(1.0, 0.25 * F), m.node_cap)
    lrs = {"alpha": node_lr * m.lr_mult, "threshold": node_lr * m.lr_mult, "leaf_logits": float(K) * m.lr_mult}
    counts = np.bincount(y, minlength=K).astype(float)
    sw = (len(y) / (K * np.maximum(counts, 1)))[y] if m.loss == "wmse" else None
    rng = np.random.default_rng(10_000 + seed)  # 공개 seed(데이터 무관)
    _, L0 = R.compute_gradients(X, Y, P, depth, approx=SPEC)
    curve, pre_max, gnorm = {}, {k: 0.0 for k in P}, {}
    for t in range(1, T + 1):
        c = 1.0 if m.c_frac == 0 else min(1.0, m.c0 + (1 - m.c0) * t / (m.c_frac * T))
        with np.errstate(all="ignore"):
            g, _ = R.compute_gradients(X, Y, P, depth, approx=SPEC, gate_scale=c, balance_lambda=m.balance,
                                       loss=m.loss, sample_weight=sw)
        if m.sigma0:
            s_t = max(0.0, 1 - t / (0.5 * T)) if m.sched == "lin" else 1 / np.sqrt(t)
        if m.lr_sched == "linear":
            mult = 1 - (t - 1) / T
        elif m.lr_sched == "cosine":
            mult = 0.5 * (1 + np.cos(np.pi * (t - 1) / T))
        elif m.lr_sched == "sqrt":
            mult = 1 / np.sqrt(t / 100 + 1)
        else:
            mult = 1.0
        for k in P:
            V[k] = 0.9 * V[k] + g[k] if t > 1 else g[k]
            step = lrs[k] * mult * V[k]
            if m.step_clip:
                b = m.step_clip * BOUNDS[k][0]
                step = clamp_plain(step, b, 1 / (4 * b))
            new = P[k] - step
            if m.sigma0:
                new = new + m.sigma0 * s_t * BOUNDS[k][0] * rng.standard_normal(new.shape)
            pre_max[k] = max(pre_max[k], float(np.abs(new).max()))
            P[k] = clamp_plain(new, *BOUNDS[k])
        if t > T - 10:
            for k in g:
                gnorm[k] = gnorm.get(k, 0.0) + float(np.linalg.norm(g[k])) / 10
        if t in (T // 3, 2 * T // 3, T):
            with np.errstate(all="ignore"):
                curve[t] = R.compute_gradients(X, Y, P, depth, approx=SPEC)[1]
    with np.errstate(all="ignore"):
        L = R.compute_gradients(X, Y, P, depth, approx=SPEC)[1]
        yh_te, _ = yhat(Xte, P, depth)
        yh_tr, _ = yhat(X, P, depth)
    return {"dataset": ds, "fold": fold, "inner": inner, "seed": seed, "method": asdict(m), "T": T, "depth": depth, "L0": float(L0), "L": float(L), "curve": curve,
            "test_acc": float((yh_te.argmax(1) == yte).mean()), "train_acc": float((yh_tr.argmax(1) == y).mean()),
            "yhat_te": yh_te.tolist(), "routing": routing(X, P, depth), "grad_norm": gnorm,
            "clamp_rate": float(np.mean(np.concatenate([(np.abs(P[k]) >= BOUNDS[k][0] - 1e-3).ravel() for k in P]))),
            "pre_clamp_max": pre_max}


def _job(args):
    return train(*args)


def methods() -> list[Method]:
    out = [Method("base")]
    out += [Method(f"B c0={c0} f={f}", c0=c0, c_frac=f) for c0, f in itertools.product([0.25, 0.5], [0.3, 0.6])]
    out += [Method(f"C s={s} {sc}", sigma0=s, sched=sc) for s, sc in itertools.product([1e-3, 3e-3, 1e-2, 3e-2], ["lin", "sqrt"])]
    out += [Method(f"D lam={l}", balance=l) for l in [0.01, 0.1, 1.0]]
    return out


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(ds, s, Method("long"), 2000) for ds in DATASETS for s in range(50)]           # L* 탐색
    jobs += [(ds, s, Method("base")) for ds in DATASETS for s in range(80)]                  # base + multi-start용
    jobs += [(ds, s, m) for ds in DATASETS for s in range(20) for m in methods()[1:]]        # B/C/D (seed 0~19, base와 짝)
    print(f"[study] {len(jobs)} runs, workers={N_WORKERS}", flush=True)
    with Pool(N_WORKERS) as pool, open(out_dir / "runs.jsonl", "w") as f:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=2), 1):
            f.write(json.dumps(r) + "\n")
            if i % 200 == 0:
                print(f"[study] {i}/{len(jobs)}", flush=True)
    print("[study] done", flush=True)


if __name__ == "__main__":
    main()
