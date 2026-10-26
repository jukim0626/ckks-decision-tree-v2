"""levelpacked의 슬롯 레이아웃/회전 로직이 reference.compute_gradients(poly 모드)와 같은 gradient를 내는지
numpy 시뮬레이터(`levelpacked/sim_engine.py`)로 검증한다. GPU/CKKS 불필요, 몇 초면 끝난다.

여러 chunk로 나뉘는 경우(breast_cancer: 노드 2개/ciphertext)와 한 ciphertext에 여러 level이 들어가는
경우(iris)를 모두 포함한다. 파라미터는 초기값이 아니라 큰 랜덤값으로 줘서 모든 경로에 실제 값이 흐르게 한다.

python -m models.gradient_soft_tree.tests.test_levelpacked_sim
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from models.gradient_soft_tree.baseline.reference import DEFAULT_POLY_SPEC, compute_gradients, predict  # noqa: E402
from models.gradient_soft_tree.levelpacked import tree as T  # noqa: E402
from models.gradient_soft_tree.levelpacked.layout import Layout  # noqa: E402
from models.gradient_soft_tree.levelpacked.sim_engine import SimContext, SimEngine  # noqa: E402

TOL = 1e-9


def _enc_data(eng, lay, X, Y=None):
    Xc = [[eng.encrypt(lay.pack_X(X, g, h)) for h in range(lay.H)] for g in range(lay.G)]
    Yc = None if Y is None else [eng.encrypt(lay.pack_Y(Y, h)) for h in range(lay.H)]
    return Xc, Yc


def check(dataset: str, depth: int, seed: int = 0, block: int = 0, groups: int = 0) -> None:
    X, Xte, y, yte, _ = load_scaled_dataset_subset(dataset)
    K = int(max(y.max(), yte.max()) + 1)
    Y = one_hot_encode(y, K)
    lay = Layout(len(y), X.shape[1], K, depth, 32768, block, groups)
    lay.validate()
    lay = lay.with_store()
    rng = np.random.default_rng(seed)
    ni = (1 << depth) - 1
    params = {"alpha": rng.normal(0, 0.8, (ni, X.shape[1])), "threshold": rng.uniform(-0.9, 0.9, (ni, X.shape[1])),
              "leaf_logits": rng.normal(0, 1.0, (1 << depth, K))}

    ctx = SimContext(SimEngine())
    eng = ctx.engine
    store = [eng.encrypt(v) for v in lay.pack_store(params)]
    Xc, Yc = _enc_data(eng, lay, X, Y)
    grads = T.gradients(ctx, lay, T.default_spec(), Xc, Yc, store)
    got = lay.unpack_store([eng.decrypt(g) for g in grads])
    ref, _ = compute_gradients(X, Y, params, depth, approx=DEFAULT_POLY_SPEC)
    err = max(np.abs(got[k] - ref[k]).max() for k in ref)

    # encrypted inference 경로(test set, 학습과 같은 block 크기/feature 그룹)
    lay_te = Layout(len(yte), X.shape[1], K, depth, 32768, lay.B, lay.G).with_store()
    Xte_c, _ = _enc_data(eng, lay_te, Xte)
    scores = [eng.decrypt(c) for c in T.predict_scores(ctx, lay_te, T.default_spec(), Xte_c, store)]
    pred = np.array([[scores[s // lay.B][c * lay.B + s % lay.B] for c in range(K)] for s in range(len(yte))]).argmax(1)
    ref_pred = predict(Xte, {**params, "steepness": 8.0, "depth": depth}, approx=DEFAULT_POLY_SPEC)
    status = "PASS" if err < TOL and (pred == ref_pred).all() else "FAIL"
    print(f"[{status}] {dataset:14s} depth={depth} {lay.describe()} grad max err={err:.2e} "
          f"predict match={(pred == ref_pred).mean():.3f}")
    assert status == "PASS"


def check_weighted(dataset: str, depth: int, kind: str, n_models: int = 1, balance: float = 0.0,
                   block: int = 0, groups: int = 0, seed: int = 0) -> None:
    """가중 MSE(class_weight, 2026-10-27): client가 암호화한 W[h] = w*2/n 경로가 reference(loss="wmse")와
    같은 gradient를 내는지. multi-start(R개 모델, 모델마다 다른 파라미터)와 balance penalty를 같이 켠 경우도 본다."""
    X, _, y, yte, _ = load_scaled_dataset_subset(dataset)
    K = int(max(y.max(), yte.max()) + 1)
    Y = one_hot_encode(y, K)
    lay = Layout(len(y), X.shape[1], K, depth, 32768, block, groups, n_models=n_models)
    lay.validate()
    lay = lay.with_store()
    rng = np.random.default_rng(seed)
    ni = (1 << depth) - 1
    plist = [{"alpha": rng.normal(0, 0.8, (ni, X.shape[1])), "threshold": rng.uniform(-0.9, 0.9, (ni, X.shape[1])),
              "leaf_logits": rng.normal(0, 1.0, (1 << depth, K))} for _ in range(n_models)]
    w = T.class_sample_weights(y, K, kind)
    Wn = np.repeat((w * (2.0 / lay.n_samples))[:, None], K, axis=1)

    ctx = SimContext(SimEngine())
    eng = ctx.engine
    store = [eng.encrypt(v) for v in lay.pack_store(plist if n_models > 1 else plist[0])]
    Xc, Yc = _enc_data(eng, lay, X, Y)
    Wc = [eng.encrypt(lay.pack_Y(Wn, h)) for h in range(lay.H)]
    grads = T.gradients(ctx, lay, T.default_spec(), Xc, Yc, store, balance=balance, W=Wc)
    got = lay.unpack_store([eng.decrypt(g) for g in grads])
    got = got if n_models > 1 else [got]
    err = 0.0
    for r in range(n_models):
        ref, _ = compute_gradients(X, Y, plist[r], depth, approx=DEFAULT_POLY_SPEC, balance_lambda=balance,
                                   loss="wmse", sample_weight=w)
        err = max(err, max(np.abs(got[r][k] - ref[k]).max() for k in ref))
    # 가중이 실제로 효과가 있는지(가중 없는 gradient와 달라야 함)
    ref_mse, _ = compute_gradients(X, Y, plist[0], depth, approx=DEFAULT_POLY_SPEC, balance_lambda=balance)
    diff = max(np.abs(got[0][k] - ref_mse[k]).max() for k in ref_mse)
    status = "PASS" if err < TOL and diff > 1e-6 else "FAIL"
    print(f"[{status}] {dataset:14s} depth={depth} wmse({kind}) R={n_models} λ={balance} {lay.describe()} "
          f"grad max err={err:.2e} (vs 가중 없음 {diff:.2e}, w max {w.max():.2f})")
    assert status == "PASS"


def main() -> None:
    for ds in ["iris", "wine", "breast_cancer", "soybean"]:
        for depth in [1, 2, 3]:
            check(ds, depth)
    # 타일을 강제로 나눈 경우(feature 그룹 x 샘플 그룹 합산 경로 검증)
    check("iris", 3, block=64, groups=1)    # H=2 (샘플 그룹만)
    check("wine", 3, block=64, groups=2)    # H=3, G=2 (둘 다)
    check("wine", 2, block=32, groups=3)    # H=5, G=3
    check("digits", 2)
    # 가중 MSE(2026-10-27)
    check_weighted("breast_cancer", 3, "inv")
    check_weighted("soybean", 3, "inv")
    check_weighted("soybean", 2, "sqrt", balance=1.0)
    check_weighted("breast_cancer_nominal", 3, "inv", n_models=2, balance=1.0)   # multi-start + balance
    check_weighted("wine", 2, "inv", n_models=4)
    check_weighted("wine", 3, "inv", block=64, groups=2)     # H=3, G=2 타일
    print("\n모든 levelpacked 시뮬레이션 검증 통과.")


if __name__ == "__main__":
    main()
