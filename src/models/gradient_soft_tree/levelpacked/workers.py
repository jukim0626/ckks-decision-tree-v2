"""levelpacked 학습/추론의 단계별 worker. GPU 메모리를 확실히 회수하려고 단계마다 별도 프로세스로 실행한다
(기존 packed와 같은 이유 - `train.py`가 subprocess로 호출).

역할 구분(1-1 원칙):
- setup    [client] 키 생성, 데이터/라벨을 levelpacked 레이아웃으로 암호화, 초기 파라미터 암호화
- iter     [server] gradient + update 1회 (sk 없이)
- finalize [client] 학습된 파라미터를 decrypt해서 정확도/에뮬레이션 대비 오차/구간 진단(검증용, protocol 아님)
- predict  [client 암호화 -> server forward -> client decrypt] test set encrypted inference

python -m models.gradient_soft_tree.levelpacked.workers <setup|iter|finalize|predict> <session_dir> [...]
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.ckks_engine import create_bootstrap_context, create_bootstrap_engine  # noqa: E402
from core.data.dataset import (  # noqa: E402
    evaluation_splits, fit_scaler, load_scaler, one_hot_encode, save_scaler, scale_features, split_dataset_subset,
)
from core.data.serialization import client_sk_path, load_client_secret_key, load_server_context, write_keys  # noqa: E402
from core.runtime.gpu import GpuPeakWatcher  # noqa: E402
from models.gradient_soft_tree.baseline.reference import (  # noqa: E402
    DEFAULT_POLY_SPEC, domain_report, init_params, predict as plain_predict,
)
from models.gradient_soft_tree.levelpacked import tree as T  # noqa: E402
from models.gradient_soft_tree.levelpacked.layout import Layout  # noqa: E402
from models.gradient_soft_tree.packed.instrumentation import CountingEngineProxy  # noqa: E402

MEASUREMENTS = "measurements.jsonl"


def _config(session_dir: Path) -> dict:
    return json.loads((session_dir / "config.json").read_text())


def _split(cfg: dict):
    """논문 실험(2026-10-08~)은 fold 번호로 `evaluation_splits`(3-fold CV / holdout)를 쓴다. fold가 없으면 예전 80/20."""
    if cfg.get("fold") is not None:
        return evaluation_splits(cfg["dataset"])[cfg["fold"]]
    X_tr_raw, X_te_raw, y_tr, y_te, _ = split_dataset_subset(cfg["dataset"], test_size=cfg["test_size"], max_train=cfg.get("max_train"))
    return X_tr_raw, X_te_raw, y_tr, y_te


def _engine(cfg: dict):
    return create_bootstrap_engine(mode="gpu", device_id=0, level_preset=cfg["level_preset"])


def _read_list(engine, d: Path, prefix: str, n: int) -> list:
    return [engine.read_ciphertext(d / f"{prefix}_{q}.ct") for q in range(n)]


def _write_list(engine, d: Path, prefix: str, cts: list) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for q, ct in enumerate(cts):
        engine.write_ciphertext(ct, d / f"{prefix}_{q}.ct")


def _init_all(F: int, K: int, depth: int, seed: int, n_models: int):
    """모델 r의 초기 seed = seed + 1000 r (r=0은 단일 모델과 같음)."""
    ps = [init_params(F, K, depth, seed + 1000 * r) for r in range(n_models)]
    return ps[0] if n_models == 1 else ps


# ------------------------------------------------------------------ setup [client]
def setup(session_dir: Path, dataset: str, depth: int, seed: int, level_preset: int, split: str, opt_json: str,
          n_models: int = 1, bootstrap_key: str = "small") -> None:
    """split: "fold:<k>"(evaluation_splits의 k번째) 또는 test_size 숫자(예전 방식)."""
    fold = int(split.split(":")[1]) if split.startswith("fold:") else None
    test_size = None if fold is not None else float(split)
    X_tr_raw, X_te_raw, y_tr, y_te = _split({"dataset": dataset, "test_size": test_size, "fold": fold})
    scaler = fit_scaler(X_tr_raw)
    X = scale_features(scaler, X_tr_raw)
    K = int(max(y_tr.max(), y_te.max()) + 1)
    lay = Layout(len(y_tr), X.shape[1], K, depth, 32768, n_models=n_models)
    lay.validate()
    lay = lay.with_store()

    times = {}
    t = time.time()
    # bootstrap_key: small(0.26GiB, ~4.4s) | regular(~9.9GiB, ~1.5s, merge 시 ct 2개에 ~1.5s) - 2026-10-09
    ctx = create_bootstrap_context(mode="gpu", level_preset=level_preset, regular_bootstrap_key=(bootstrap_key == "regular"))
    times["keygen_s"] = time.time() - t
    session_dir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    write_keys(ctx, session_dir / "keys", sk_path=client_sk_path(session_dir))
    times["write_keys_s"] = time.time() - t
    save_scaler(scaler, session_dir / "client" / "scaler.json", dataset_name=dataset)
    eng = ctx.engine
    data_dir = session_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    Y = one_hot_encode(y_tr, K)
    t = time.time()
    for h in range(lay.H):  # (feature 그룹 g, 샘플 그룹 h) 타일마다 ciphertext 하나 - client가 평문으로 배치 후 암호화
        for g in range(lay.G):
            eng.write_ciphertext(eng.encrypt(lay.pack_X(X, g, h).tolist(), ctx.pk), data_dir / f"X_{g}_{h}.ct")
        eng.write_ciphertext(eng.encrypt(lay.pack_Y(Y, h).tolist(), ctx.pk), data_dir / f"Y_{h}.ct")
    times["encrypt_train_data_s"] = time.time() - t  # 패킹 + 암호화 + 저장(학습 데이터 X, Y)
    params0 = _init_all(X.shape[1], K, depth, seed, n_models)
    t = time.time()
    # 파라미터는 저장 형태(store가 여러 개면 압축 compact)로 암호화 - Layout.pack_params
    _write_list(eng, session_dir / "params", "p", [eng.encrypt(v.tolist(), ctx.pk) for v in lay.pack_params(params0)])
    times["encrypt_init_params_s"] = time.time() - t
    print(f"[setup] layout: {lay.describe()}", flush=True)
    cfg = {"setup_times": times, "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
           "n_data_ct": lay.G * lay.H + lay.H, "n_param_ct": lay.n_params,
           "n_internal": (1 << depth) - 1, "n_leaves": 1 << depth, "layout_desc": lay.describe(), "dataset": dataset, "depth": depth, "seed": seed, "level_preset": level_preset, "test_size": test_size, "fold": fold,
           "layout": lay.to_dict(), "opt": json.loads(opt_json), "bootstrap_key": bootstrap_key, "n_store": lay.n_store, "iterations_done": 0,
           "dataset": dataset}
    (session_dir / "config.json").write_text(json.dumps(cfg))


def _n_boot(counter) -> int:
    """bootstrap 호출 수(merge_bootstrap 1회 = ct 2개를 한 번에 = 호출 1회로 셈. ct 수는 refreshed_ct)."""
    return counter.counts.get("bootstrap", 0) + counter.counts.get("merge_bootstrap", 0)


# ------------------------------------------------------------------ iter [server]
def iterate(session_dir: Path) -> None:
    t_proc = time.time()
    cfg = _config(session_dir)
    lay = Layout.from_dict(cfg["layout"])
    opt = T.OptConfig(**cfg["opt"])
    engine = _engine(cfg)
    counter = CountingEngineProxy(engine)
    ctx = load_server_context(counter, session_dir / "keys", mode="gpu", device_id=0)
    X = [[engine.read_ciphertext(session_dir / "data" / f"X_{g}_{h}.ct") for h in range(lay.H)] for g in range(lay.G)]
    Y = [engine.read_ciphertext(session_dir / "data" / f"Y_{h}.ct") for h in range(lay.H)]
    params = _read_list(engine, session_dir / "params", "p", lay.n_params)
    vel_dir = session_dir / "velocity"
    velocity = _read_list(engine, vel_dir, "vel", lay.n_params) if vel_dir.exists() else None

    t0 = time.time()
    with GpuPeakWatcher() as gpu:
        store = T.materialize(ctx, lay, params)
        grads = T.gradients(ctx, lay, T.default_spec(), X, Y, store, balance=opt.balance)
        t_grad = time.time() - t0
        n_boot_grad = _n_boot(counter)
        params, velocity = T.update(ctx, lay, opt, params, grads, velocity)
    t_total = time.time() - t0

    _write_list(engine, session_dir / "params", "p", params)
    if velocity is not None:
        _write_list(engine, vel_dir, "vel", velocity)
    cfg["iterations_done"] += 1
    (session_dir / "config.json").write_text(json.dumps(cfg))
    rec = {"iteration": cfg["iterations_done"], "t_grad": t_grad, "t_update": t_total - t_grad, "t_total": t_total,
           "t_process": time.time() - t_proc, "bootstrap_grad": n_boot_grad, "layout": lay.describe(), "compact": lay.use_compact,
           "bootstrap_update": _n_boot(counter) - n_boot_grad,
           "refreshed_ct": counter.n_refreshed_ciphertexts, "bootstrap_key": cfg.get("bootstrap_key", "small"), "counts": dict(counter.counts),
           "peak_gpu_mib": gpu.peak}
    # --- 검증 전용(protocol 아님, 시간 측정에서 제외): 매 iteration 파라미터 상태 확인 ---
    # 2026-10-09 wine fold 0이 100~150 iteration 사이 CKKS에서만 1e80으로 폭발(에뮬레이션은 정상) - 언제/어디서 깨지는지 잡기 위함.
    try:
        t_h = time.time()
        sk = load_client_secret_key(engine, session_dir)
        dec = lay.unpack_params([np.real(engine.decrypt(ct, sk)) for ct in params])
        rec["health"] = {k: float(np.abs(v).max()) for k, v in dec.items()}
        rec["health"]["finite"] = bool(all(np.isfinite(v).all() for v in dec.values()))
        if velocity is not None:
            vv = [np.real(engine.decrypt(ct, sk)) for ct in velocity]
            rec["health"]["velocity_max"] = float(max(np.abs(v).max() for v in vv))
        rec["health"]["t_check_s"] = time.time() - t_h
    except Exception as e:  # 검증 실패가 학습을 멈추게 하지는 않는다
        rec["health"] = {"error": str(e)[:200]}
    with open(session_dir / MEASUREMENTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("iteration", "t_grad", "t_update", "bootstrap_grad", "bootstrap_update", "peak_gpu_mib")}))


# ------------------------------------------------------------------ finalize [client, 검증용]
def finalize(session_dir: Path) -> None:
    cfg = _config(session_dir)
    lay = Layout.from_dict(cfg["layout"])
    opt = T.OptConfig(**cfg["opt"])
    engine = _engine(cfg)
    sk = load_client_secret_key(engine, session_dir)
    params = lay.unpack_params([np.real(engine.decrypt(ct, sk)) for ct in _read_list(engine, session_dir / "params", "p", lay.n_params)])

    X_tr_raw, X_te_raw, y_tr, y_te = _split(cfg)
    scaler = load_scaler(session_dir / "client" / "scaler.json", expected_dataset_name=cfg["dataset"], expected_n_features=lay.n_features)
    X, Xte = scale_features(scaler, X_tr_raw), scale_features(scaler, X_te_raw)
    R = lay.n_models
    plist = params if R > 1 else [params]
    fulls = [{**p, "steepness": 8.0, "depth": lay.depth} for p in plist]
    full = fulls if R > 1 else fulls[0]
    n_it = cfg["iterations_done"]
    inits = _init_all(lay.n_features, lay.n_classes, lay.depth, cfg["seed"], R)
    inits = inits if R > 1 else [inits]
    emus = [T.emulate(X, one_hot_encode(y_tr, lay.n_classes), lay.depth, T.default_spec(), opt, n_it, p0)[-1] for p0 in inits]
    emu_full = [{**e, "steepness": 8.0, "depth": lay.depth} for e in emus]
    result = {
        "iterations": n_it,
        "train_acc_poly": float((plain_predict(X, full, DEFAULT_POLY_SPEC) == y_tr).mean()),
        "test_acc_poly": float((plain_predict(Xte, full, DEFAULT_POLY_SPEC) == y_te).mean()),
        "test_acc_true": float((plain_predict(Xte, full, "true") == y_te).mean()),
        "emulation_test_acc_poly": float((plain_predict(Xte, emu_full if R > 1 else emu_full[0], DEFAULT_POLY_SPEC) == y_te).mean()),
        "max_err_vs_emulation": {k: float(max(np.abs(p[k] - e[k]).max() for p, e in zip(plist, emus))) for k in plist[0]},
        "domain_train": max((domain_report(X, f) for f in fulls), key=lambda d: d["max_abs_gate_input"]),
        "domain_test": max((domain_report(Xte, f) for f in fulls), key=lambda d: d["max_abs_gate_input"]),
        "n_models": R,
        "per_model_test_acc_poly": [float((plain_predict(Xte, f, DEFAULT_POLY_SPEC) == y_te).mean()) for f in fulls],
    }
    print(json.dumps(result))


# ------------------------------------------------------------------ predict
def predict(session_dir: Path) -> None:
    cfg = _config(session_dir)
    lay = Layout.from_dict(cfg["layout"])
    engine = _engine(cfg)
    ctx = load_server_context(engine, session_dir / "keys", mode="gpu", device_id=0)
    # [client] test set을 학습과 같은 block 크기의 레이아웃으로 암호화
    _, X_te_raw, _, y_te = _split(cfg)
    scaler = load_scaler(session_dir / "client" / "scaler.json", expected_dataset_name=cfg["dataset"], expected_n_features=lay.n_features)
    Xte = scale_features(scaler, X_te_raw)
    lay_te = Layout(len(y_te), lay.n_features, lay.n_classes, lay.depth, lay.slot_count, lay.B, lay.G, lay.n_models).with_store()
    t0 = time.time()
    Xc = [[engine.encrypt(lay_te.pack_X(Xte, g, h).tolist(), ctx.pk) for h in range(lay_te.H)] for g in range(lay_te.G)]
    t_enc = time.time() - t0
    # [server] 파라미터를 decrypt하지 않고 forward
    t0 = time.time()
    params = _read_list(engine, session_dir / "params", "p", lay.n_params)
    t_load = time.time() - t0
    t0 = time.time()
    store = T.materialize(ctx, lay, params)
    t_expand = time.time() - t0
    t0 = time.time()
    scores_ct = T.predict_scores(ctx, lay_te, T.default_spec(), Xc, store)
    t_eval = time.time() - t0
    t = t_expand + t_eval
    # [client] 최종 점수만 decrypt
    sk = load_client_secret_key(engine, session_dir)
    t0 = time.time()
    v = [np.real(engine.decrypt(ct, sk)) for ct in scores_ct]
    t_dec = time.time() - t0
    scores = np.array([[v[s // lay.B][c * lay.B + s % lay.B] for c in range(lay.n_classes)] for s in range(len(y_te))])
    pred = scores.argmax(1)
    n = int(len(y_te))
    print(json.dumps({"encrypted_test_acc": float((pred == y_te).mean()), "n_test": n,
                      "t_forward": t,                       # server: 파라미터 펼치기 + forward (예전 키 이름 유지)
                      "t_encrypt_test_s": t_enc, "t_load_params_s": t_load, "t_expand_params_s": t_expand,
                      "t_eval_s": t_eval, "t_decrypt_s": t_dec,
                      "t_server_per_sample_s": t / n,       # 배치(SIMD) 처리 시간을 샘플 수로 나눈 환산값
                      "n_test_ct": lay_te.G * lay_te.H}))


def main() -> None:
    cmd, session_dir = sys.argv[1], Path(sys.argv[2])
    if cmd == "setup":
        setup(session_dir, sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7], sys.argv[8],
              int(sys.argv[9]) if len(sys.argv) > 9 else 1, sys.argv[10] if len(sys.argv) > 10 else "small")
    elif cmd == "iter":
        iterate(session_dir)
    elif cmd == "finalize":
        finalize(session_dir)
    elif cmd == "predict":
        predict(session_dir)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
