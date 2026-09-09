"""depth1_reference.py의 forward/backward/SGD update를 실제 CKKS 암호문 연산으로 그대로
옮긴 것 - MGI/Gini 계산이 없는 gradient soft tree를 처음으로 암호화 상태에서 검증하는
스크립트.

**설계 원칙**: alpha(feature attention)/threshold/leaf_logits는 전부 **private
training data로부터 학습되는 값**이므로 (closed_form_mgi의 threshold/soft-MGI weight와
같은 지위), 학습 내내 한 번도 decrypt하지 않고 ciphertext로만 갱신한다 - 이 프로젝트의
"client decrypt 없음" 원칙을 gradient descent 계보에도 그대로 적용.

**재사용한 공용 프리미티브**(2026-09-09 리팩터로 core/ 패키지로 이동됨): `sigmoid_approx_enc`,
`ensure_level`/`create_bootstrap_context`, `scatter_to_slot`/`next_power_of_two`,
`packed_softmax`/`softmax_backward_packed`(bounded Newton-Raphson reciprocal 기반 일반
softmax, MGI z-score 특화 로직 없음 - exp 근사는 alpha/threshold/leaf_logits가 학습 중
실제로 도달하는 범위를 plaintext로 먼저 측정해서 interval=[-2.5,2.5]로 피팅한 버전).

**패딩 슬롯 처리**: threshold는 전체 슬롯에 broadcast된 ciphertext라(=`ctx.engine.sum()`의
출력 형태와 동일), `sigmoid_approx_enc(enc_feature-threshold)`가 padding 슬롯
(n_samples..slot_count-1)에서도 0이 아닌 값을 만든다. 샘플 축으로 합산
(`ctx.engine.sum`)하기 직전마다 `sample_mask`(1로 채워진 첫 n_samples 슬롯, 나머지
0)를 곱해서 이 오염을 제거한다 - 매 sum 호출 전에 한 번씩.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.ckks_engine import create_bootstrap_context, ensure_level  # noqa: E402
from core.data.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from core.encrypted_ops.slot_packing import extract_weight_broadcast as _extract_weight_broadcast, next_power_of_two, scatter_to_slot  # noqa: E402
from core.approximation.sigmoid import STEEPNESS, sigmoid_approx_enc  # noqa: E402
from core.encrypted_ops.softmax import packed_softmax, softmax_backward_packed  # noqa: E402
from models.gradient_soft_tree.depth1_reference import (  # noqa: E402
    predict as plaintext_predict,
    softmax,
    softmax_backward,
    train_depth1 as plaintext_train_depth1,
)

# depth=3 CKKS 학습이 CUDA OOM나는 근본 원인이 "파라미터의 level이 낮을수록 bootstrap이
# 훨씬 자주 걸리고, desilofhe는 프로세스 안에서 GPU 메모리를 절대 안 돌려주니 bootstrap
# 총 호출 횟수가 곧 peak 메모리"라는 걸 실측으로 확인함(EXPERIMENT_LOG 2026-08-26 참고).
_LOCAL_MIN_LEVEL = 5


def _ensure_level(ctx, ct):
    return ensure_level(ctx, ct, min_level=_LOCAL_MIN_LEVEL)


def init_encrypted_params(ctx, n_features: int, n_classes: int, seed: int, slot_count: int):
    """plaintext_reference와 완전히 같은 초기화(같은 seed의 numpy RNG)를 암호화해서 대응
    시켜야 두 트랙 비교가 의미 있다."""
    rng = np.random.default_rng(seed)
    alpha0 = rng.normal(0, 0.1, size=n_features)
    threshold0 = rng.normal(0, 0.1, size=n_features)
    leaf_L0 = rng.normal(0, 0.1, size=n_classes)
    leaf_R0 = rng.normal(0, 0.1, size=n_classes)

    n_pow2_f = next_power_of_two(n_features)
    n_pow2_c = next_power_of_two(n_classes)

    alpha_vec = alpha0.tolist() + [0.0] * (n_pow2_f - n_features)
    alpha_ct = ctx.engine.encrypt(alpha_vec, ctx.pk)

    threshold_cts = [ctx.engine.encrypt([float(threshold0[j])] * slot_count, ctx.pk) for j in range(n_features)]

    leaf_L_vec = leaf_L0.tolist() + [0.0] * (n_pow2_c - n_classes)
    leaf_R_vec = leaf_R0.tolist() + [0.0] * (n_pow2_c - n_classes)
    leaf_L_ct = ctx.engine.encrypt(leaf_L_vec, ctx.pk)
    leaf_R_ct = ctx.engine.encrypt(leaf_R_vec, ctx.pk)

    return {
        "alpha": alpha_ct,
        "threshold": threshold_cts,
        "leaf_L": leaf_L_ct,
        "leaf_R": leaf_R_ct,
        "plaintext_init": {"alpha": alpha0, "threshold": threshold0, "leaf_L": leaf_L0, "leaf_R": leaf_R0},
    }


def forward_backward_update(ctx, dataset, params: dict, sample_mask, n_features: int, n_classes: int, lr: float):
    n_pow2_f = next_power_of_two(n_features)
    n_pow2_c = next_power_of_two(n_classes)

    w = packed_softmax(ctx, params["alpha"], n_features, n_pow2_f)  # (n_pow2_f slots)

    gate_terms = []  # gate_j ciphertext per feature (per-sample packed) - backward에서 재사용
    gate = None
    for j in range(n_features):
        enc_feature = _ensure_level(ctx, dataset.enc_features[j])
        threshold_j = _ensure_level(ctx, params["threshold"][j])
        enc_diff = ctx.engine.subtract(enc_feature, threshold_j)
        gate_j = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff)
        gate_terms.append(gate_j)

        w_j = _extract_weight_broadcast(ctx, w, j)
        piece = ctx.engine.multiply(w_j, gate_j, ctx.rlk)
        gate = piece if gate is None else ctx.engine.add(gate, piece)
        gc.collect()
    gate = _ensure_level(ctx, gate)

    left_prob = ctx.engine.subtract(1.0, gate)
    right_prob = gate

    leafdist_L = packed_softmax(ctx, params["leaf_L"], n_classes, n_pow2_c)
    leafdist_R = packed_softmax(ctx, params["leaf_R"], n_classes, n_pow2_c)

    y_hat = []
    for c in range(n_classes):
        ld_L_c = _ensure_level(ctx, _extract_weight_broadcast(ctx, leafdist_L, c))
        ld_R_c = _ensure_level(ctx, _extract_weight_broadcast(ctx, leafdist_R, c))
        term_L = ctx.engine.multiply(left_prob, ld_L_c, ctx.rlk)
        term_R = ctx.engine.multiply(right_prob, ld_R_c, ctx.rlk)
        y_hat.append(_ensure_level(ctx, ctx.engine.add(term_L, term_R)))

    n_samples = dataset.n_samples
    dL_dyhat = []
    for c in range(n_classes):
        diff = ctx.engine.subtract(y_hat[c], dataset.enc_labels[c])
        dL_dyhat.append(ctx.engine.multiply(diff, 2.0 / n_samples))

    dL_dleafdist_L_packed = None
    dL_dleafdist_R_packed = None
    dL_dgate = None
    for c in range(n_classes):
        dL_dyhat_c = _ensure_level(ctx, dL_dyhat[c])
        term_L = _ensure_level(ctx, ctx.engine.multiply(dL_dyhat_c, left_prob, ctx.rlk))
        term_L = ctx.engine.multiply(term_L, sample_mask, ctx.rlk)
        term_L = ctx.engine.intt(term_L)
        sum_L = _ensure_level(ctx, ctx.engine.sum(term_L, ctx.rotation_key))
        piece_L = scatter_to_slot(ctx, sum_L, c)
        dL_dleafdist_L_packed = piece_L if dL_dleafdist_L_packed is None else ctx.engine.add(dL_dleafdist_L_packed, piece_L)

        term_R = _ensure_level(ctx, ctx.engine.multiply(dL_dyhat_c, right_prob, ctx.rlk))
        term_R = ctx.engine.multiply(term_R, sample_mask, ctx.rlk)
        term_R = ctx.engine.intt(term_R)
        sum_R = _ensure_level(ctx, ctx.engine.sum(term_R, ctx.rotation_key))
        piece_R = scatter_to_slot(ctx, sum_R, c)
        dL_dleafdist_R_packed = piece_R if dL_dleafdist_R_packed is None else ctx.engine.add(dL_dleafdist_R_packed, piece_R)

        # dL/dgate = sum_c dL_dyhat_c * (leafdist_R_c - leafdist_L_c)
        ld_L_c = _extract_weight_broadcast(ctx, leafdist_L, c)
        ld_R_c = _extract_weight_broadcast(ctx, leafdist_R, c)
        coef = _ensure_level(ctx, ctx.engine.subtract(ld_R_c, ld_L_c))
        piece_gate = ctx.engine.multiply(dL_dyhat_c, coef, ctx.rlk)
        dL_dgate = piece_gate if dL_dgate is None else ctx.engine.add(dL_dgate, piece_gate)
    dL_dgate = _ensure_level(ctx, dL_dgate)
    dL_dleafdist_L_packed = _ensure_level(ctx, dL_dleafdist_L_packed)
    dL_dleafdist_R_packed = _ensure_level(ctx, dL_dleafdist_R_packed)

    dL_dleaf_L = softmax_backward_packed(ctx, leafdist_L, dL_dleafdist_L_packed, n_classes, n_pow2_c)
    dL_dleaf_R = softmax_backward_packed(ctx, leafdist_R, dL_dleafdist_R_packed, n_classes, n_pow2_c)

    dL_dt = []
    dL_dalpha_packed = None
    for j in range(n_features):
        gate_j = _ensure_level(ctx, gate_terms[j])
        surrogate = ctx.engine.multiply(gate_j, ctx.engine.subtract(1.0, gate_j), ctx.rlk)  # gate_j*(1-gate_j)
        surrogate = _ensure_level(ctx, surrogate)
        w_j = _ensure_level(ctx, _extract_weight_broadcast(ctx, w, j))

        # threshold gradient: -steepness * w_j * sum_samples(dL_dgate * surrogate)
        prod_t = ctx.engine.multiply(dL_dgate, surrogate, ctx.rlk)
        prod_t = _ensure_level(ctx, prod_t)
        prod_t = ctx.engine.multiply(prod_t, sample_mask, ctx.rlk)
        prod_t = ctx.engine.intt(prod_t)
        sum_t = _ensure_level(ctx, ctx.engine.sum(prod_t, ctx.rotation_key))
        dL_dt_j = _ensure_level(ctx, ctx.engine.multiply(sum_t, w_j, ctx.rlk))
        dL_dt_j = ctx.engine.multiply(dL_dt_j, -STEEPNESS)
        dL_dt.append(_ensure_level(ctx, dL_dt_j))

        # alpha gradient term: sum_samples(dL_dgate * (gate_j - gate))
        term_a = ctx.engine.multiply(dL_dgate, ctx.engine.subtract(gate_j, gate), ctx.rlk)
        term_a = _ensure_level(ctx, term_a)
        term_a = ctx.engine.multiply(term_a, sample_mask, ctx.rlk)
        term_a = ctx.engine.intt(term_a)
        sum_a = _ensure_level(ctx, ctx.engine.sum(term_a, ctx.rotation_key))
        piece_a = scatter_to_slot(ctx, sum_a, j)
        dL_dalpha_packed = piece_a if dL_dalpha_packed is None else ctx.engine.add(dL_dalpha_packed, piece_a)
        gc.collect()

    dL_dalpha_packed = _ensure_level(ctx, dL_dalpha_packed)
    w = _ensure_level(ctx, w)
    dL_dalpha = ctx.engine.multiply(w, dL_dalpha_packed, ctx.rlk)

    new_params = {
        "alpha": _ensure_level(ctx, ctx.engine.subtract(params["alpha"], ctx.engine.multiply(dL_dalpha, lr))),
        "threshold": [
            _ensure_level(ctx, ctx.engine.subtract(params["threshold"][j], ctx.engine.multiply(dL_dt[j], lr)))
            for j in range(n_features)
        ],
        "leaf_L": _ensure_level(ctx, ctx.engine.subtract(params["leaf_L"], ctx.engine.multiply(dL_dleaf_L, lr))),
        "leaf_R": _ensure_level(ctx, ctx.engine.subtract(params["leaf_R"], ctx.engine.multiply(dL_dleaf_R, lr))),
    }
    return new_params


def decrypt_params(ctx, params: dict, n_features: int, n_classes: int) -> dict:
    alpha = np.real(ctx.engine.decrypt(params["alpha"], ctx.sk))[:n_features]
    threshold = np.array([np.real(ctx.engine.decrypt(t, ctx.sk))[0] for t in params["threshold"]])
    leaf_L = np.real(ctx.engine.decrypt(params["leaf_L"], ctx.sk))[:n_classes]
    leaf_R = np.real(ctx.engine.decrypt(params["leaf_R"], ctx.sk))[:n_classes]
    return {"alpha": alpha, "threshold": threshold, "leaf_L": leaf_L, "leaf_R": leaf_R, "steepness": STEEPNESS}


def main():
    """단일 프로세스로 몇 epoch만 빠르게 검증하는 디버그 진입점 - **실측으로 epoch 3에서
    desilofhe GPU 메모리 누적(EXPERIMENT_LOG.md 2026-08-11/12와 동일 원인)으로 CUDA OOM이
    남**. 여러 epoch을 실제로 돌리려면 `train_depth1_ckks.py`(epoch마다 별도 프로세스로
    격리하는 오케스트레이터, node_worker.py/depth_worker.py와 같은 패턴)를 쓸 것."""
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "iris"
    n_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    lr = 0.3
    seed = 0

    X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name)
    n_features = X_train.shape[1]
    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    y_train_oh = one_hot_encode(y_train, n_classes)

    print(f"[setup] dataset={dataset_name} n_samples={X_train.shape[0]} n_features={n_features} n_classes={n_classes}")
    ctx = create_bootstrap_context(mode="gpu")
    dataset = encrypt_dataset(ctx, X_train, y_train_oh)
    sample_mask = ctx.engine.encrypt([1.0] * dataset.n_samples, ctx.pk)

    params = init_encrypted_params(ctx, n_features, n_classes, seed=seed, slot_count=ctx.engine.slot_count)
    plaintext_init = params.pop("plaintext_init")

    print("[reference] running plaintext trajectory for the same epoch counts...")
    ref_params_by_epoch = {}
    for k in range(n_epochs + 1):
        ref_params_by_epoch[k] = plaintext_train_depth1(X_train, y_train_oh, steepness=STEEPNESS, lr=lr, epochs=k, seed=seed)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        params = forward_backward_update(ctx, dataset, params, sample_mask, n_features, n_classes, lr=lr)
        elapsed = time.time() - t0
        decoded = decrypt_params(ctx, params, n_features, n_classes)
        ref = ref_params_by_epoch[epoch]
        print(f"\n[epoch {epoch}] elapsed={elapsed:.1f}s")
        print(f"  alpha     ckks={np.round(decoded['alpha'],4)}  ref={np.round(ref['alpha'],4)}")
        print(f"  threshold ckks={np.round(decoded['threshold'],4)}  ref={np.round(ref['threshold'],4)}")
        print(f"  leaf_L    ckks={np.round(decoded['leaf_L'],4)}  ref={np.round(ref['leaf_L'],4)}")
        print(f"  leaf_R    ckks={np.round(decoded['leaf_R'],4)}  ref={np.round(ref['leaf_R'],4)}")
        max_err = max(
            np.abs(decoded['alpha']-ref['alpha']).max(),
            np.abs(decoded['threshold']-ref['threshold']).max(),
            np.abs(decoded['leaf_L']-ref['leaf_L']).max(),
            np.abs(decoded['leaf_R']-ref['leaf_R']).max(),
        )
        print(f"  max abs diff vs plaintext reference = {max_err:.5f}")
        gc.collect()


if __name__ == "__main__":
    main()
