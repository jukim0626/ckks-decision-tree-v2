"""ROADMAP B: 다항식 근사 구간을 지키면서 적은 iteration으로 수렴하는 학습법 탐색 (plaintext, CPU).

배경: CKKS에서 iteration 하나가 수백 초라 iteration 수가 귀하다(빨리 수렴해야 함). 그런데
빨리 학습시키면(momentum, 큰 lr) 파라미터가 sigmoid/exp 다항식의 유효 구간 밖으로 나가고,
CKKS에서는 그 순간 계산이 망가진다. 이 스크립트는 여러 학습 설정을 **CKKS와 같은 다항식**
(`reference.PolySpec`)으로 학습시켜서 정확도와 구간 이탈 여부를 함께 잰다.

탐색 축:
- spec: 다항식 구간. cur = 지금 CKKS(sigmoid [-2,2] 15차, exp [-2.5,2.5] 20차),
        wide = sigmoid [-3,3] 31차 + exp [-4,4] 20차(+Newton 반복 12회)
- opt: gd(지금) / mom(heavy-ball momentum) / nest(Nesterov, lookahead 없는 형태) - 둘 다 CKKS에선
       파라미터당 velocity ciphertext 1개 + 덧셈/스칼라곱만 추가
- lr, kappa: 노드 파라미터(alpha/threshold) lr = lr * max(1, kappa * n_features)
             (gate가 softmax(alpha) 가중합이라 노드 gradient가 ~1/F로 작아지는 걸 보정)
- tclip: threshold를 |t| <= sigmoid 구간 - 1로 제한(x가 [-1,1]이라 이러면 x-t가 절대 구간을 안 넘음),
         alpha/leaf_logits도 |v| <= exp 구간으로 제한.
         none / hard(np.clip - CKKS에선 직접 불가, 효과 상한 확인용) / poly(31차 Chebyshev clamp 다항식,
         CKKS에서 실제로 가능한 형태 - 매 iteration 적용하면 근사 오차가 누적되는 부작용이 있음) /
         comp(합성 다항식 sign 기반 clamp, CKKS 가능 + 누적 오차 없음)
- center: 매 iteration 후 alpha/leaf_logits에서 클래스(또는 feature) 평균을 뺀다. softmax는
          상수 shift에 불변이라 모델 함수는 그대로이고 값의 범위만 0 근처로 유지된다
          (CKKS 비용: rotation sum 1번 + 뺄셈)

구간 판정: sigmoid 입력은 엄격(구간을 조금만 넘어도 poly(2.2)=-4.9처럼 폭발), softmax 입력은
구간+1.0까지 허용(exp 다항식은 구간 밖 1.0에서도 상대오차 ~3e-4로 정확 - 분모 Newton 발산 같은
진짜 문제는 poly 에뮬레이션이 그대로 재현하므로 정확도에 이미 반영됨).

평가 규약(test 누수 방지): train을 다시 fit/val(75/25)로 나눠 fit으로 학습, val로 설정을 고르고,
보고는 test로 한다. scaler는 fit 부분으로만 맞춘다. 구간 이탈은 매 iteration 학습 데이터 기준으로
검사하고, 한 번이라도 벗어나면 그 run은 unsafe로 표시한다(CKKS였다면 그 시점에 계산이 깨짐).

사용법(src/에서):
  python -m models.gradient_soft_tree.experiments.domain_safe_optim sweep  <out.jsonl> [iterations] [seeds]
  python -m models.gradient_soft_tree.experiments.domain_safe_optim summary <out.jsonl>
  python -m models.gradient_soft_tree.experiments.domain_safe_optim confirm <out.jsonl> [seeds]
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
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.approximation.exp import chebyshev_approximation_exp_plain  # noqa: E402
from core.approximation.sigmoid import STEEPNESS, chebyshev_approximation  # noqa: E402
from core.data.dataset import fit_scaler, one_hot_encode, scale_features, split_dataset_subset  # noqa: E402
from models.gradient_soft_tree.baseline.reference import (  # noqa: E402
    DEFAULT_POLY_SPEC,
    PolySpec,
    compute_gradients,
    init_params,
    predict,
)

DATASETS = ["iris", "wine", "breast_cancer", "soybean"]
DEPTH = 3
DIVERGED = 1e6  # 파라미터가 이 크기를 넘으면 발산으로 보고 중단

SPECS = {
    "cur": DEFAULT_POLY_SPEC,
    "wide": PolySpec(
        sigmoid_coeffs=tuple(chebyshev_approximation(31, 3.0, STEEPNESS)),
        sigmoid_interval=3.0,
        exp_coeffs=tuple(chebyshev_approximation_exp_plain(20, (-4.0, 4.0))),
        exp_interval=4.0,
        recip_iterations=12,
    ),
    # 2026-10-07: steepness 개념 제거 실험 - 기본 sigmoid σ(x-t)를 [-2,2]에서 7차로 근사(오차 3e-5).
    # exp 쪽은 cur와 동일.
    "s1": PolySpec(
        sigmoid_coeffs=tuple(chebyshev_approximation(7, 2.0, 1.0)),
        sigmoid_interval=2.0,
        steepness=1.0,
    ),
}


@dataclass(frozen=True)
class Config:
    spec: str = "cur"
    opt: str = "gd"        # gd | mom | nest
    beta: float = 0.0
    lr: float = 2.0
    kappa: float = 0.0     # 0이면 노드 lr 보정 없음
    center: bool = False
    tclip: str = "none"    # none | hard | poly

    def name(self) -> str:
        o = self.opt if self.opt == "gd" else f"{self.opt}{self.beta}"
        return f"{self.spec}|{o}|lr{self.lr}|k{self.kappa}|{'C' if self.center else '-'}|t{self.tclip}"


def load_split(dataset: str):
    """train -> fit/val(75/25), test는 그대로. scaler는 fit 부분으로만 맞추고 전부 clip 적용."""
    X_tr_raw, X_te_raw, y_tr, y_te, _ = split_dataset_subset(dataset, test_size=0.2)
    X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
        X_tr_raw, y_tr, test_size=0.25, random_state=0, stratify=y_tr
    )
    scaler = fit_scaler(X_fit_raw)
    n_classes = int(max(y_tr.max(), y_te.max()) + 1)
    return {
        "X_fit": scale_features(scaler, X_fit_raw), "y_fit": y_fit,
        "X_val": scale_features(scaler, X_val_raw), "y_val": y_val,
        "X_test": scale_features(scaler, X_te_raw), "y_test": y_te,
        "n_classes": n_classes,
    }


SOFTMAX_MARGIN = 1.0


def _domain_ok(X: np.ndarray, params: dict, spec: PolySpec) -> tuple[bool, float, float]:
    """(구간 안인가, max|x-t|, max|softmax 입력|)."""
    gate_in = float(np.abs(X[:, None, :] - params["threshold"][None, :, :]).max())
    soft_in = float(max(np.abs(params["alpha"]).max(), np.abs(params["leaf_logits"]).max()))
    return (gate_in <= spec.sigmoid_interval and soft_in <= spec.exp_interval + SOFTMAX_MARGIN), gate_in, soft_in


def _clamp_poly(bound: float, width: float, degree: int = 31):
    """[-width, width]에서 clamp(x, -bound, bound)를 근사하는 Chebyshev 다항식(CKKS로 평가 가능)."""
    from numpy.polynomial import chebyshev as C
    k = np.arange(degree + 1)
    xs = np.cos((2 * k + 1) * np.pi / (2 * (degree + 1))) * width
    coef = C.chebfit(xs / width, np.clip(xs, -bound, bound), degree)
    return lambda x: C.chebval(x / width, coef)


def _clamp_composite(bound: float, radius: float, k: int = 12):
    """합성 다항식으로 만든 clamp(x, -bound, bound). |x| <= radius - bound에서 유효.

    sign(z) ~= f^k(z/radius), f(y) = (3y - y^3)/2 (HE 비교 연산에서 쓰는 표준 합성 다항식)
    |z| = z * sign(z),  min(x, B) = (x + B - |x - B|)/2,  max(y, -B) = (y - B + |y + B|)/2.
    f는 3차라 1회당 레벨 2개 -> sign 1번 ~2k 레벨, clamp 1번에 sign 2번. 단일 고차 clamp
    다항식과 달리 구간 안에서 거의 정확한 항등함수라 매 iteration 적용해도 값이 끌려가지 않는다."""
    def sign(z):
        y = z / radius
        for _ in range(k):
            y = (3 * y - y ** 3) / 2
        return y

    def clamp(x):
        z = x - bound
        y = (x + bound - z * sign(z)) / 2
        z = y + bound
        return (y - bound + z * sign(z)) / 2

    return clamp


COMP_MARGIN = 0.05

# threshold: |t| <= sigmoid 구간-1 (폭: sigmoid 구간), alpha/leaf: |v| <= exp 구간 (폭: exp 구간+1)
_CLAMPS = {
    name: {
        "threshold": _clamp_poly(spec.sigmoid_interval - 1.0, spec.sigmoid_interval),
        "softmax": _clamp_poly(spec.exp_interval, spec.exp_interval + SOFTMAX_MARGIN),
        # 합성 clamp는 경계에서 ~0.01 넘칠 수 있어 경계를 COMP_MARGIN만큼 안쪽으로 둔다
        "threshold_comp": _clamp_composite(spec.sigmoid_interval - 1.0 - COMP_MARGIN, 4.0 * spec.sigmoid_interval),
        "softmax_comp": _clamp_composite(spec.exp_interval - COMP_MARGIN, 4.0 * (spec.exp_interval + SOFTMAX_MARGIN)),
    }
    for name, spec in SPECS.items()
}


def train(cfg: Config, X: np.ndarray, y: np.ndarray, n_classes: int, iterations: int, seed: int) -> dict:
    spec = SPECS[cfg.spec]
    Y = one_hot_encode(y, n_classes)
    n_features = X.shape[1]
    params = init_params(n_features, n_classes, DEPTH, seed)
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    node_lr = cfg.lr * max(1.0, cfg.kappa * n_features)
    lrs = {"alpha": node_lr, "threshold": node_lr, "leaf_logits": cfg.lr}

    safe, max_gate_in, max_soft_in, first_violation, diverged = True, 0.0, 0.0, None, False
    for it in range(1, iterations + 1):
        grads, _loss = compute_gradients(X, Y, params, DEPTH, spec.steepness, spec)
        for k in params:
            g = grads[k]
            if cfg.opt == "gd":
                step = g
            else:
                velocity[k] = cfg.beta * velocity[k] + g
                step = velocity[k] if cfg.opt == "mom" else g + cfg.beta * velocity[k]
            params[k] = params[k] - lrs[k] * step
        # 2026-10-07 2차 sweep: threshold만 clip하면 alpha/leaf 쪽 이탈이 남아서 셋 다 clip한다.
        t_bound, s_bound = spec.sigmoid_interval - 1.0, spec.exp_interval
        if cfg.tclip == "hard":
            params["threshold"] = np.clip(params["threshold"], -t_bound, t_bound)
            params["alpha"] = np.clip(params["alpha"], -s_bound, s_bound)
            params["leaf_logits"] = np.clip(params["leaf_logits"], -s_bound, s_bound)
        elif cfg.tclip == "poly":
            params["threshold"] = _CLAMPS[cfg.spec]["threshold"](params["threshold"])
            params["alpha"] = _CLAMPS[cfg.spec]["softmax"](params["alpha"])
            params["leaf_logits"] = _CLAMPS[cfg.spec]["softmax"](params["leaf_logits"])
        elif cfg.tclip == "comp":
            params["threshold"] = _CLAMPS[cfg.spec]["threshold_comp"](params["threshold"])
            params["alpha"] = _CLAMPS[cfg.spec]["softmax_comp"](params["alpha"])
            params["leaf_logits"] = _CLAMPS[cfg.spec]["softmax_comp"](params["leaf_logits"])
        if cfg.center:
            params["alpha"] = params["alpha"] - params["alpha"].mean(axis=1, keepdims=True)
            params["leaf_logits"] = params["leaf_logits"] - params["leaf_logits"].mean(axis=1, keepdims=True)

        ok, gi, si = _domain_ok(X, params, spec)
        max_gate_in, max_soft_in = max(max_gate_in, gi), max(max_soft_in, si)
        if not ok and safe:
            safe, first_violation = False, it
        if not np.isfinite(gi + si) or max(gi, si) > DIVERGED:
            diverged = True
            break
    return {
        "params": params, "safe": safe, "first_violation": first_violation, "diverged": diverged,
        "max_gate_in": max_gate_in, "max_softmax_in": max_soft_in,
    }


def _acc(X, y, params, spec) -> float:
    with np.errstate(all="ignore"):
        pred = predict(X, {**params, "steepness": spec.steepness, "depth": DEPTH}, approx=spec)
    return float((pred == y).mean())


def run_one(args) -> dict:
    cfg, dataset, iterations, seed = args
    d = _DATA[dataset]
    with np.errstate(all="ignore"):
        r = train(cfg, d["X_fit"], d["y_fit"], d["n_classes"], iterations, seed)
    spec = SPECS[cfg.spec]
    out = {
        "config": asdict(cfg), "name": cfg.name(), "dataset": dataset, "iterations": iterations, "seed": seed,
        "safe": r["safe"], "first_violation": r["first_violation"], "diverged": r["diverged"],
        "max_gate_in": r["max_gate_in"], "max_softmax_in": r["max_softmax_in"],
    }
    if r["diverged"]:
        out.update(val_acc=None, test_acc=None, fit_acc=None)
    else:
        p = r["params"]
        out.update(
            fit_acc=_acc(d["X_fit"], d["y_fit"], p, spec),
            val_acc=_acc(d["X_val"], d["y_val"], p, spec),
            test_acc=_acc(d["X_test"], d["y_test"], p, spec),
            test_domain_ok=_domain_ok(d["X_test"], p, spec)[0],
        )
    return out


_DATA: dict = {}


def _init_worker():
    for ds in DATASETS:
        _DATA[ds] = load_split(ds)


def config_grid() -> list[Config]:
    cfgs = []
    opts = [("gd", 0.0), ("mom", 0.5), ("mom", 0.9), ("nest", 0.9)]
    # center는 1차 sweep(2026-10-07)에서 효과 없음 확인(이탈의 대부분이 threshold 쪽) - 축에서 제외
    for spec, (opt, beta), lr, kappa, tclip in itertools.product(
        SPECS, opts, [1.0, 2.0, 4.0, 6.0, 8.0], [0.0, 0.25, 0.5], ["none", "hard", "poly", "comp"]
    ):
        cfgs.append(Config(spec, opt, beta, lr, kappa, False, tclip))
    return cfgs


def sweep(out_path: Path, iterations: int, seeds: list[int]) -> None:
    jobs = [(c, ds, iterations, s) for c in config_grid() for ds in DATASETS for s in seeds]
    print(f"[sweep] {len(jobs)} runs ({len(config_grid())} configs x {len(DATASETS)} datasets x {len(seeds)} seeds)", flush=True)
    with Pool(processes=min(20, os.cpu_count() or 4), initializer=_init_worker) as pool, open(out_path, "w") as f:
        for i, res in enumerate(pool.imap_unordered(run_one, jobs, chunksize=4), 1):
            f.write(json.dumps(res) + "\n")
            if i % 500 == 0:
                print(f"[sweep] {i}/{len(jobs)}", flush=True)


def summary(out_path: Path) -> None:
    rows = [json.loads(l) for l in open(out_path)]
    by = {}
    for r in rows:
        by.setdefault((r["name"], r["dataset"]), []).append(r)

    def agg(name, ds):
        rs = by[(name, ds)]
        safe = all(r["safe"] and not r["diverged"] and r.get("test_domain_ok", False) for r in rs)
        val = np.mean([r["val_acc"] if r["val_acc"] is not None else 0.0 for r in rs])
        test = [r["test_acc"] if r["test_acc"] is not None else 0.0 for r in rs]
        return safe, float(val), float(np.mean(test)), float(np.std(test))

    names = sorted({r["name"] for r in rows})
    baseline = {"iris": "cur|gd|lr2.0|k0.0|-|tnone", "wine": "cur|gd|lr6.0|k0.0|-|tnone",
                "breast_cancer": "cur|gd|lr2.0|k0.0|-|tnone", "soybean": "cur|gd|lr6.0|k0.0|-|tnone"}
    print("\n== 데이터셋별: 현재 방식 vs val로 고른 최선(구간 안전한 설정만) ==")
    for ds in DATASETS:
        b = agg(baseline[ds], ds)
        cands = [(agg(n, ds), n) for n in names if agg(n, ds)[0]]
        best = max(cands, key=lambda x: (x[0][1], -len(x[1])))
        n_unsafe = sum(1 for n in names if not agg(n, ds)[0])
        print(f"{ds:14s} 현재 {baseline[ds]:30s} test={b[2]:.3f}±{b[3]:.3f} safe={b[0]} | "
              f"최선 {best[1]:34s} val={best[0][1]:.3f} test={best[0][2]:.3f}±{best[0][3]:.3f} | unsafe 설정 {n_unsafe}/{len(names)}")

    print("\n== 데이터셋 공통 설정(전부 구간 안전, val 평균 최대) top 10 ==")
    glob = []
    for n in names:
        a = [agg(n, ds) for ds in DATASETS]
        if all(x[0] for x in a):
            glob.append((np.mean([x[1] for x in a]), n, a))
    glob.sort(reverse=True)
    for v, n, a in glob[:10]:
        print(f"{n:34s} val_mean={v:.3f} | " + " ".join(f"{ds[:5]}={x[2]:.3f}" for ds, x in zip(DATASETS, a)))

    print("\n== 축별 영향: 구간 이탈(unsafe) 비율 ==")
    for key in ["spec", "opt", "tclip", "kappa"]:
        groups = {}
        for (name, ds), rs in by.items():
            cfg = rs[0]["config"]
            k = cfg[key] if key != "opt" else (cfg["opt"] if cfg["opt"] == "gd" else f"{cfg['opt']}{cfg['beta']}")
            groups.setdefault(k, []).append(agg(name, ds)[0])
        print(f"{key:7s} " + " | ".join(f"{k}: unsafe {1 - np.mean(v):.0%}" for k, v in sorted(groups.items(), key=str)))


CONFIRM_CONFIGS = {
    "현재(GD, 데이터셋별 lr)": None,  # 데이터셋별 baseline lr 사용
    "mom0.9 lr1 k0.25 comp": Config("cur", "mom", 0.9, 1.0, 0.25, False, "comp"),
    "nest0.9 lr2 comp": Config("cur", "nest", 0.9, 2.0, 0.0, False, "comp"),
    "wide mom0.5 lr4 comp": Config("wide", "mom", 0.5, 4.0, 0.0, False, "comp"),
}
BASELINE_LR = {"iris": 2.0, "wine": 6.0, "breast_cancer": 2.0, "soybean": 6.0}


def _ceiling_job(args):
    """plaintext 상한선: 진짜 sigmoid/exp, vanilla GD, 구간 제약 없음, lr은 val로 선택."""
    from models.gradient_soft_tree.baseline.reference import train_depthN
    dataset, iterations, seed = args
    d = _DATA[dataset]
    Y = one_hot_encode(d["y_fit"], d["n_classes"])
    best = None
    for lr in [1.0, 2.0, 4.0, 6.0]:
        p = train_depthN(d["X_fit"], Y, depth=DEPTH, lr=lr, epochs=iterations, seed=seed, approx="true")
        val = float((predict(d["X_val"], p, "true") == d["y_val"]).mean())
        if best is None or val > best[0]:
            best = (val, float((predict(d["X_test"], p, "true") == d["y_test"]).mean()))
    return {"label": f"상한선(true GD {iterations}it)", "dataset": dataset, "seed": seed, "test_acc": best[1], "safe": None}


def _confirm_job(args):
    label, cfg, dataset, iterations, seed = args
    if cfg is None:
        cfg = Config("cur", "gd", 0.0, BASELINE_LR[dataset])
    r = run_one((cfg, dataset, iterations, seed))
    safe = r["safe"] and not r["diverged"] and r.get("test_domain_ok", False)
    return {"label": f"{label} @{iterations}it", "dataset": dataset, "seed": seed, "test_acc": r["test_acc"] or 0.0, "safe": safe}


def confirm(out_path: Path, seeds: list[int]) -> None:
    jobs = [(lab, cfg, ds, it, s) for lab, cfg in CONFIRM_CONFIGS.items() for ds in DATASETS
            for it in [30, 50, 100] for s in seeds]
    ceil_jobs = [(ds, it, s) for ds in DATASETS for it in [300] for s in seeds]
    with Pool(processes=min(28, os.cpu_count() or 4), initializer=_init_worker) as pool:
        res = pool.map(_confirm_job, jobs) + pool.map(_ceiling_job, ceil_jobs)
    with open(out_path, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    labels = list(dict.fromkeys(r["label"] for r in res))
    print(f"\n== test 정확도 (seed {len(seeds)}개 평균±표준편차, [unsafe]=한 seed라도 구간 이탈) ==")
    print(f"{'설정':34s} " + " ".join(f"{ds:>15s}" for ds in DATASETS))
    for lab in labels:
        cells = []
        for ds in DATASETS:
            rs = [r for r in res if r["label"] == lab and r["dataset"] == ds]
            acc = [r["test_acc"] for r in rs]
            flag = "!" if any(r["safe"] is False for r in rs) else " "
            cells.append(f"{np.mean(acc):.3f}±{np.std(acc):.3f}{flag}")
        print(f"{lab:34s} " + " ".join(f"{c:>15s}" for c in cells))
    print("(! = unsafe: CKKS에서는 구간 이탈로 계산이 깨질 수 있음)")


def main() -> None:
    cmd, out = sys.argv[1], Path(sys.argv[2])
    if cmd == "sweep":
        iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        seeds = list(range(int(sys.argv[4]) if len(sys.argv) > 4 else 3))
        sweep(out, iterations, seeds)
        summary(out)
    elif cmd == "confirm":
        confirm(out, list(range(int(sys.argv[3]) if len(sys.argv) > 3 else 5)))
    elif cmd == "summary":
        summary(out)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
