"""ROADMAP B 2단계 전 검증: domain_safe_optim(plaintext 에뮬레이션)에서 고른 새 연산들이 실제 CKKS에서도
에뮬레이션과 같은 결과를 내는지 확인한다.

1) clamp: 합성 다항식 clamp를 CKKS로 반복 적용(학습 중 매 iteration 적용과 같은 상황)하면서 plaintext와
   비교. 경계 근처 값(sign 반복이 노이즈를 키울 수 있는 곳)과 범위를 크게 넘는 값을 일부러 포함한다.
2) train: momentum + 노드 lr 보정 + 합성 clamp projection을 넣은 짧은 CKKS 학습. 매 iteration 뒤 파라미터를
   (검증용으로) decrypt해서 같은 규칙의 plaintext 에뮬레이션과 비교한다.
   gradient는 기존 `forward_backward_update_N_packed`를 lr=1로 불러 g = old - new로 얻는다(학습 코드 무수정).
   검증 전용 도구라 iteration마다 client sk로 decrypt한다 - protocol 일부 아님.

사용법(src/에서, GPU):
  python -m models.gradient_soft_tree.experiments.validate_domain_safe_ckks clamp [repeats]
  python -m models.gradient_soft_tree.experiments.validate_domain_safe_ckks train <dataset> <depth> <iterations>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.ckks_engine import create_bootstrap_context, ensure_level  # noqa: E402
from models.gradient_soft_tree.experiments.domain_safe_optim import _clamp_composite  # noqa: E402
from models.gradient_soft_tree.packed.instrumentation import CountingEngineProxy  # noqa: E402

SIGN_K = 12
_F_COEFFS = [0.0, 1.5, 0.0, -0.5]  # f(y) = (3y - y^3)/2


def he_sign(ctx, z, radius: float, k: int = SIGN_K):
    """sign(z) ~= f^k(z / radius). f는 3차라 1회 레벨 ~2."""
    y = ctx.engine.multiply(z, 1.0 / radius)
    for _ in range(k):
        y = ensure_level(ctx, y, min_level=5)
        y = ctx.engine.evaluate_polynomial(y, _F_COEFFS, ctx.rlk)
    return y


def he_clamp(ctx, x, bound: float, radius: float, k: int = SIGN_K):
    """clamp(x, -bound, bound) = max(min(x, B), -B),
    min(x, B) = (x + B - |x - B|)/2,  max(y, -B) = (y - B + |y + B|)/2,  |z| = z * sign(z).
    `domain_safe_optim._clamp_composite`의 CKKS 버전(연산 순서 동일)."""
    eng = ctx.engine
    x = ensure_level(ctx, x, min_level=5)
    z = eng.subtract(x, bound)
    s = ensure_level(ctx, he_sign(ctx, z, radius, k), min_level=5)
    abs_z = eng.multiply(ensure_level(ctx, z, min_level=5), s, ctx.rlk)
    y = eng.multiply(eng.subtract(eng.add(x, bound), abs_z), 0.5)
    y = ensure_level(ctx, y, min_level=5)
    z = eng.add(y, bound)
    s = ensure_level(ctx, he_sign(ctx, z, radius, k), min_level=5)
    abs_z = eng.multiply(ensure_level(ctx, z, min_level=5), s, ctx.rlk)
    out = eng.multiply(eng.add(eng.subtract(y, bound), abs_z), 0.5)
    return ensure_level(ctx, out, min_level=5)


def test_clamp(repeats: int) -> None:
    ctx = create_bootstrap_context(mode="gpu", level_preset=17)
    counter = CountingEngineProxy(ctx.engine)
    ctx.engine = counter

    # (bound, radius): threshold용(cur spec), softmax 입력용(cur spec) - domain_safe_optim과 동일
    cases = {"threshold": (0.95, 8.0), "softmax": (2.45, 14.0)}
    for name, (bound, radius) in cases.items():
        rng = np.random.default_rng(0)
        vals = np.concatenate([
            np.linspace(-bound, bound, 41),                     # 구간 안
            bound + np.array([-0.05, -0.01, 0.0, 0.01, 0.05]),  # 경계 바로 근처(+)
            -bound + np.array([-0.05, -0.01, 0.0, 0.01, 0.05]), # 경계 바로 근처(-)
            rng.uniform(-(radius - bound), radius - bound, 40), # 범위 밖까지 큰 값
        ])
        n = len(vals)
        plain_clamp = _clamp_composite(bound, radius, SIGN_K)
        ct = ctx.engine.encrypt(vals.tolist(), ctx.pk)
        plain = vals.copy()
        counter.counts.clear()
        t0 = time.time()
        print(f"\n[{name}] bound={bound} radius={radius} k={SIGN_K}, {n}개 값")
        for r in range(1, repeats + 1):
            ct = he_clamp(ctx, ct, bound, radius)
            plain = plain_clamp(plain)
            dec = np.real(ctx.engine.decrypt(ct, ctx.sk))[:n]
            if r in (1, 2, 5, 10, 20, 30) or r == repeats:
                err = np.abs(dec - plain)
                print(f"  repeat {r:2d}: max|CKKS-plain|={err.max():.2e} (경계근처 {err[41:51].max():.2e}) "
                      f"max|CKKS|={np.abs(dec).max():.4f} (<= {bound}?) "
                      f"max|CKKS-np.clip|={np.abs(dec - np.clip(vals, -bound, bound)).max():.3e}", flush=True)
        el = time.time() - t0
        print(f"  clamp 1회 평균 {el / repeats:.2f}s, bootstrap {counter.counts.get('bootstrap', 0) / repeats:.1f}회/clamp")


# ---- 2) 짧은 학습 비교 ----
# domain_safe_optim의 제안 설정: Config("cur", "mom", 0.9, 1.0, 0.25, False, "comp")
BETA, LR, KAPPA = 0.9, 1.0, 0.25
THR_BOUND, THR_RADIUS = 2.0 - 1.0 - 0.05, 4.0 * 2.0     # domain_safe_optim._CLAMPS["cur"]["threshold_comp"]
SM_BOUND, SM_RADIUS = 2.5 - 0.05, 4.0 * (2.5 + 1.0)     # domain_safe_optim._CLAMPS["cur"]["softmax_comp"]
HISTORY = "validate_history.jsonl"


def _node_lr(n_features: int) -> float:
    return LR * max(1.0, KAPPA * n_features)


def train_worker(session_dir: Path) -> None:
    """한 iteration(서버 측 계산) + 검증용 decrypt 기록(client 측)."""
    import json
    from dataclasses import replace

    from core.ckks_engine import create_bootstrap_engine
    from core.data.serialization import load_client_secret_key, load_server_context, read_dataset
    from models.gradient_soft_tree.baseline.tree_ops import decrypt_params_N
    from models.gradient_soft_tree.packed.block_ops import (
        assert_layout_fits, build_block_masks, compute_block_size_tight, pack_dataset_features_blocked,
        sample_mask_blocked_plain, sample_mask_plain,
    )
    from models.gradient_soft_tree.packed.tree_ops_packed import forward_backward_update_N_packed
    from models.gradient_soft_tree.params import (
        load_alpha, load_leaf_logits, load_threshold, save_alpha, save_leaf_logits, save_threshold,
    )

    config = json.loads((session_dir / "config.json").read_text())
    n_features, n_classes, depth = config["n_features"], config["n_classes"], config["depth"]
    n_internal, n_leaves = (1 << depth) - 1, 1 << depth
    engine = create_bootstrap_engine(mode=config["mode"], device_id=config["device_id"], level_preset=config["level_preset"])
    ctx = load_server_context(engine, session_dir / "keys", mode=config["mode"], device_id=config["device_id"])
    dataset = read_dataset(ctx, session_dir / "dataset", n_features=n_features, n_classes=n_classes, n_samples=config["n_samples"])
    block_size = compute_block_size_tight(dataset.n_samples)
    assert_layout_fits(n_features, block_size, engine.slot_count)
    params_dir, vel_dir = session_dir / "params", session_dir / "velocity"
    params = {
        "alpha": load_alpha(engine, params_dir, n_internal),
        "threshold": load_threshold(engine, params_dir, n_internal, n_features),
        "leaf_logits": load_leaf_logits(engine, params_dir, n_leaves),
    }
    t0 = time.time()
    # gradient: lr=1로 한 번 갱신한 결과와의 차이(g = old - new). 학습 코드는 그대로 재사용.
    stepped = forward_backward_update_N_packed(
        ctx, dataset, params, sample_mask_plain(dataset.n_samples, engine.slot_count),
        pack_dataset_features_blocked(ctx, dataset.enc_features, block_size),
        sample_mask_blocked_plain(dataset.n_samples, n_features, block_size, engine.slot_count),
        build_block_masks(n_features, block_size, engine.slot_count), block_size, n_features, n_classes, depth, lr=1.0,
    )
    t_grad = time.time() - t0
    has_vel = vel_dir.exists()
    vel = {
        "alpha": load_alpha(engine, vel_dir, n_internal) if has_vel else None,
        "threshold": load_threshold(engine, vel_dir, n_internal, n_features) if has_vel else None,
        "leaf_logits": load_leaf_logits(engine, vel_dir, n_leaves) if has_vel else None,
    }
    node_lr = _node_lr(n_features)

    def update(old, new, v_old, lr, bound, radius):
        g = engine.subtract(old, new)
        v = g if v_old is None else engine.add(engine.multiply(v_old, BETA), g)
        v = ensure_level(ctx, v, min_level=5)
        p = engine.subtract(ensure_level(ctx, old, min_level=5), engine.multiply(v, lr))
        return he_clamp(ctx, p, bound, radius), v

    new_params, new_vel = {"alpha": [], "threshold": [], "leaf_logits": []}, {"alpha": [], "threshold": [], "leaf_logits": []}
    for i in range(n_internal):
        p, v = update(params["alpha"][i], stepped["alpha"][i], vel["alpha"] and vel["alpha"][i], node_lr, SM_BOUND, SM_RADIUS)
        new_params["alpha"].append(p); new_vel["alpha"].append(v)
        row_p, row_v = [], []
        for j in range(n_features):
            p, v = update(params["threshold"][i][j], stepped["threshold"][i][j],
                          vel["threshold"] and vel["threshold"][i][j], node_lr, THR_BOUND, THR_RADIUS)
            row_p.append(p); row_v.append(v)
        new_params["threshold"].append(row_p); new_vel["threshold"].append(row_v)
    for l in range(n_leaves):
        p, v = update(params["leaf_logits"][l], stepped["leaf_logits"][l], vel["leaf_logits"] and vel["leaf_logits"][l], LR, SM_BOUND, SM_RADIUS)
        new_params["leaf_logits"].append(p); new_vel["leaf_logits"].append(v)
    t_total = time.time() - t0

    save_alpha(engine, params_dir, new_params["alpha"]); save_threshold(engine, params_dir, new_params["threshold"])
    save_leaf_logits(engine, params_dir, new_params["leaf_logits"])
    vel_dir.mkdir(exist_ok=True)
    save_alpha(engine, vel_dir, new_vel["alpha"]); save_threshold(engine, vel_dir, new_vel["threshold"])
    save_leaf_logits(engine, vel_dir, new_vel["leaf_logits"])

    # --- 검증 전용(client 측): 이번 iteration 결과를 decrypt해서 기록 ---
    cctx = replace(ctx, sk=load_client_secret_key(engine, session_dir))
    dec = decrypt_params_N(cctx, new_params, n_features, n_classes, depth)
    with open(session_dir / HISTORY, "a") as f:
        f.write(json.dumps({k: np.asarray(dec[k]).tolist() for k in ("alpha", "threshold", "leaf_logits")}
                           | {"t_grad": t_grad, "t_total": t_total}) + "\n")


def train_compare(dataset_name: str, depth: int, iterations: int) -> None:
    import json

    from core.data.dataset import load_scaler, one_hot_encode, scale_features, split_dataset_subset
    from core.runtime.gpu import wait_for_gpu_settle
    from core.runtime.session import make_session_dir
    from core.runtime.subprocess_worker import run_worker_module
    from models.gradient_soft_tree.baseline.reference import compute_gradients, init_params

    session_dir = make_session_dir(f"validate_domain_safe_d{depth}_")
    print(f"[validate] {dataset_name} depth={depth} iterations={iterations} session={session_dir}", flush=True)
    wait_for_gpu_settle()
    run_worker_module("models.gradient_soft_tree.baseline.setup_worker", str(session_dir), dataset_name, str(depth),
                      "0", str(LR), "17", "none", "0.2")
    for it in range(1, iterations + 1):
        wait_for_gpu_settle()
        run_worker_module("models.gradient_soft_tree.experiments.validate_domain_safe_ckks", "worker", str(session_dir))
        print(f"[validate] iteration {it}/{iterations} done", flush=True)

    # 같은 규칙의 plaintext 에뮬레이션(같은 train set, 같은 scaler, 같은 seed 초기화)
    config = json.loads((session_dir / "config.json").read_text())
    X_raw, _, y, _, _ = split_dataset_subset(dataset_name, test_size=0.2, max_train=None)
    X = scale_features(load_scaler(session_dir / "client" / "scaler.json", expected_dataset_name=dataset_name,
                                   expected_n_features=config["n_features"]), X_raw)
    Y = one_hot_encode(y, config["n_classes"])
    params = init_params(config["n_features"], config["n_classes"], depth, seed=0)
    vel = {k: np.zeros_like(v) for k, v in params.items()}
    lrs = {"alpha": _node_lr(config["n_features"]), "threshold": _node_lr(config["n_features"]), "leaf_logits": LR}
    clamps = {"alpha": _clamp_composite(SM_BOUND, SM_RADIUS, SIGN_K), "leaf_logits": _clamp_composite(SM_BOUND, SM_RADIUS, SIGN_K),
              "threshold": _clamp_composite(THR_BOUND, THR_RADIUS, SIGN_K)}
    history = [json.loads(l) for l in open(session_dir / HISTORY)]
    print("\n iter | max|CKKS-emul| alpha / threshold / leaf | max|threshold| | iter time(s) grad+update")
    for it, h in enumerate(history, 1):
        grads, _ = compute_gradients(X, Y, params, depth, approx="poly")
        for k in params:
            vel[k] = BETA * vel[k] + grads[k]
            params[k] = clamps[k](params[k] - lrs[k] * vel[k])
        errs = [np.abs(np.array(h[k]) - params[k]).max() for k in ("alpha", "threshold", "leaf_logits")]
        print(f" {it:4d} | {errs[0]:.2e} / {errs[1]:.2e} / {errs[2]:.2e} | {np.abs(np.array(h['threshold'])).max():.4f} | "
              f"{h['t_grad']:.0f}+{h['t_total'] - h['t_grad']:.0f}", flush=True)
    print(f"[validate] session_dir={session_dir}")


def bench_sign(level_preset, repeats: int = 3) -> None:
    """clamp 비용 절감 후보 비교: 합성 다항식 sign(f^12) vs desilofhe sign_bootstrap(bootstrap 1회에 sign 계산).
    sign_bootstrap은 입력이 [-1,1]로 정규화돼 있어야 한다(archive/closed_form_mgi 기록) - clamp는 z/radius를 넣는다.
    level_preset 17 엔진은 sign_bootstrap을 지원하지 않아(2026-10-08 확인) 26-level(None)도 함께 잰다."""
    ctx = create_bootstrap_context(mode="gpu", level_preset=level_preset)
    eng = ctx.engine
    n = 2001
    z = np.linspace(-1, 1, n)
    ct = eng.encrypt(z.tolist(), ctx.pk)
    keys = (ctx.rlk, ctx.conjugation_key, ctx.rotation_key, ctx.small_bootstrap_key)
    print(f"\n== level_preset={level_preset} (fresh level {ct.level}) ==")

    def timed(name, fn):
        try:
            fn()  # warm-up
            t0 = time.time()
            for _ in range(repeats):
                out = fn()
            t = (time.time() - t0) / repeats
            print(f"  {name}: {t:.2f}s/회, 출력 level {out.level}")
            return out
        except RuntimeError as e:
            print(f"  {name}: 지원 안 함 ({str(e)[:60]})")
            return None

    timed("bootstrap", lambda: eng.bootstrap(eng.intt(ct), *keys))
    timed("lossy_bootstrap", lambda: eng.lossy_bootstrap(eng.intt(ct), *keys))
    sb = timed("sign_bootstrap", lambda: eng.sign_bootstrap(eng.intt(ct), *keys))
    comp = timed(f"합성 sign(k={SIGN_K})", lambda: he_sign(ctx, ct, 1.0, SIGN_K))
    for name, out in [("sign_bootstrap", sb), ("합성 sign", comp)]:
        if out is None:
            continue
        d = np.real(eng.decrypt(out, ctx.sk))[:n]
        errs = " ".join(f"|z|>={lo}:{np.abs(d[np.abs(z) >= lo] - np.sign(z[np.abs(z) >= lo])).max():.1e}" for lo in [0.01, 0.05, 0.1])
        print(f"  {name} sign 오차 {errs} | |z| 근사 max err {np.abs(z * d - np.abs(z)).max():.2e}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "clamp"
    if cmd == "clamp":
        test_clamp(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    elif cmd == "signbench":
        for preset in (17, None):
            bench_sign(preset)
    elif cmd == "worker":
        train_worker(Path(sys.argv[2]))
    elif cmd == "train":
        train_compare(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
