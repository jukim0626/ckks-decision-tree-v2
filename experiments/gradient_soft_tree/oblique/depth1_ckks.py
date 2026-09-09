"""depth1_ckks.py(axis-aligned attention-blend gate)와 완전히 같은 구조/프리미티브를
재사용하되, gate 계산만 oblique(선형결합+sigmoid 1번)로 바꾼 버전. `oblique/depthN_reference.py`
(plaintext)의 CKKS 이식판 - 딱 depth=1만 검증하는 최소 버전.

**바뀐 부분만 요약** (나머지는 depth1_ckks.py와 100% 동일한 코드를 그대로 복사):
- 기존: alpha(softmax attention) + threshold(feature별) 두 파라미터, gate = sum_j softmax(alpha)_j * sigmoid(feature_j - threshold_j)
- 신규: w(feature별 선형 가중치) + b(스칼라) 두 파라미터, gate = sigmoid(steepness*(sum_j w_j*feature_j + b))
  -> packed_softmax(alpha) 호출이 완전히 사라짐(gate 쪽에서는). leaf softmax(leafdist_L/R)는 그대로 유지.

**backward도 마찬가지로 딱 한 단계만 다르다** (dL_dgate까지는 depth1_ckks.py와 동일한 유도):
  surrogate = gate*(1-gate)
  dL_dz     = steepness * dL_dgate * surrogate
  dL_dw_j   = sum_samples(dL_dz * feature_j)   <- 기존의 "threshold gradient + alpha gradient(softmax backward 포함)" 두 갈래가 이 한 줄로 합쳐짐
  dL_db     = sum_samples(dL_dz)
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ckks_tree import sigmoid_approx_enc  # noqa: E402
from client_assisted.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from closed_form_mgi.primitives import create_bootstrap_context, ensure_level  # noqa: E402
from closed_form_mgi.simd_argmin import next_power_of_two, scatter_to_slot  # noqa: E402
from closed_form_mgi.soft_mgi import _extract_weight_broadcast  # noqa: E402
from experiments.gradient_soft_tree.depth1_ckks import packed_softmax, softmax_backward_packed  # noqa: E402
from experiments.gradient_soft_tree.oblique.depthN_reference import train_depthN_oblique  # noqa: E402
from sigmoid_approx_coeffs import chebyshev_approximation  # noqa: E402

_LOCAL_MIN_LEVEL = 5


def _ensure_level(ctx, ct):
    return ensure_level(ctx, ct, min_level=_LOCAL_MIN_LEVEL)


STEEPNESS = 8.0  # depth1_ckks.py와 동일한 상수, sigmoid_approx_enc 계수에 이미 baked-in
# 2026-09-08: fit interval [-5,5] 확장 시도 결과가 애매해서 폐기, [-2,2](axis-aligned와
# 동일한 SPLIT_SIGMOID_COEFFS)로 되돌림. depthN_ckks.py와 같은 이유 - 코드는 참고용으로만 남김.
_WIDE_SIGMOID_COEFFS = chebyshev_approximation(degree=15, interval=5.0, steepness=STEEPNESS).tolist()

# 2026-09-08 논문 기반 검증 Phase 1 (TEL, arXiv 2002.07772 - 저차수 다항식이 구간 밖에서
# 훨씬 덜 폭발한다는 아이디어): interval은 baseline과 동일하게 [-2,2] 유지, degree만 15->5로
# 낮춤. numpy 사전분석(GPU 없이)으로 z=14에서 baseline(deg15)은 poly=-1.58e15인데 deg5는
# poly=42277로 11자릿수 작음을 확인 - 구간 내 오차는 커지지만(0.028->0.133) "터질 때 얼마나
# 크게 터지는지"가 압도적으로 줄어듦. 단, CLAUDE.md에 문서화된 "bootstrap 자체가 값 2~5
# 넘으면 조용히 깨진다"는 별개 제약이 있어서, z=ensure_level(...) 호출 시점에 z가 이미
# 크면 다항식과 무관하게 bootstrap이 먼저 깨뜨릴 수 있음 - 이 변수 하나만 테스트해서
# "다항식 차수 문제"와 "bootstrap 크기 제약 문제"를 가른다.
_LOW_DEGREE_SIGMOID_COEFFS = chebyshev_approximation(degree=5, interval=2.0, steepness=STEEPNESS).tolist()


def init_encrypted_params(ctx, n_features: int, n_classes: int, seed: int, slot_count: int):
    """plaintext oblique reference(train_depthN_oblique)와 완전히 같은 순서로 rng를 소비해야
    두 트랙의 초기값이 정확히 일치한다 - w -> b -> leaf_logits 순서(depthN_reference.py의
    draw 순서 그대로, depth=1이라 n_internal=1, n_leaves=2)."""
    rng = np.random.default_rng(seed)
    w0 = rng.normal(0, 0.1 / np.sqrt(n_features), size=(1, n_features))[0]  # 안정성 수정 #1
    b0 = rng.normal(0, 0.1, size=(1,))[0]
    leaf_logits0 = rng.normal(0, 0.1, size=(2, n_classes))
    leaf_L0, leaf_R0 = leaf_logits0[0], leaf_logits0[1]

    n_pow2_c = next_power_of_two(n_classes)

    w_cts = [ctx.engine.encrypt([float(w0[j])] * slot_count, ctx.pk) for j in range(n_features)]
    b_ct = ctx.engine.encrypt([float(b0)] * slot_count, ctx.pk)

    leaf_L_vec = leaf_L0.tolist() + [0.0] * (n_pow2_c - n_classes)
    leaf_R_vec = leaf_R0.tolist() + [0.0] * (n_pow2_c - n_classes)
    leaf_L_ct = ctx.engine.encrypt(leaf_L_vec, ctx.pk)
    leaf_R_ct = ctx.engine.encrypt(leaf_R_vec, ctx.pk)

    return {
        "w": w_cts,
        "b": b_ct,
        "leaf_L": leaf_L_ct,
        "leaf_R": leaf_R_ct,
        "plaintext_init": {"w": w0, "b": b0, "leaf_L": leaf_L0, "leaf_R": leaf_R0},
    }


def forward_backward_update(ctx, dataset, params: dict, sample_mask, n_features: int, n_classes: int, lr: float, weight_decay: float = 0.01):
    n_pow2_c = next_power_of_two(n_classes)

    # ---- forward: gate 계산만 다름 (선형결합 -> sigmoid 1번) ----
    z = None
    for j in range(n_features):
        enc_feature = _ensure_level(ctx, dataset.enc_features[j])
        w_j = _ensure_level(ctx, params["w"][j])
        term = ctx.engine.multiply(enc_feature, w_j, ctx.rlk)
        z = term if z is None else ctx.engine.add(z, term)
        gc.collect()
    b_ct = _ensure_level(ctx, params["b"])
    z = ctx.engine.add(z, b_ct)
    z = ensure_level(ctx, z, min_level=12)  # depthN_ckks.py와 동일 이유 - sigmoid poly 진입 전 별도 guard
    # 2026-09-08 Phase 1 검증: 저차수(degree=5) 게이트 다항식 단독 테스트 (interval은
    # baseline과 동일 [-2,2]) - 아래 주석 처리된 줄로 되돌리면 baseline(degree=15) 비교 가능.
    gate = sigmoid_approx_enc(ctx.engine, ctx.rlk, z, coeffs=_LOW_DEGREE_SIGMOID_COEFFS)

    left_prob = ctx.engine.subtract(1.0, gate)
    right_prob = gate

    # ---- leaf 쪽은 depth1_ckks.py와 완전히 동일 ----
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

    # ---- 여기부터 gate backward - 바뀐 부분 ----
    surrogate = ctx.engine.multiply(gate, ctx.engine.subtract(1.0, gate), ctx.rlk)  # gate*(1-gate)
    surrogate = _ensure_level(ctx, surrogate)
    dL_dz = ctx.engine.multiply(dL_dgate, surrogate, ctx.rlk)
    dL_dz = ctx.engine.multiply(dL_dz, STEEPNESS)
    dL_dz = _ensure_level(ctx, dL_dz)

    dL_dw = []
    for j in range(n_features):
        enc_feature = _ensure_level(ctx, dataset.enc_features[j])
        prod = ctx.engine.multiply(dL_dz, enc_feature, ctx.rlk)
        prod = ctx.engine.multiply(prod, sample_mask, ctx.rlk)
        prod = ctx.engine.intt(prod)
        sum_j = _ensure_level(ctx, ctx.engine.sum(prod, ctx.rotation_key))
        dL_dw.append(sum_j)
        gc.collect()

    prod_b = ctx.engine.multiply(dL_dz, sample_mask, ctx.rlk)
    prod_b = ctx.engine.intt(prod_b)
    dL_db = _ensure_level(ctx, ctx.engine.sum(prod_b, ctx.rotation_key))

    # 안정성 수정 #2,#3: gate 전용 lr_gate=lr/sqrt(n_features) + L2 weight decay
    lr_gate = lr / np.sqrt(n_features)
    new_w = []
    for j in range(n_features):
        decay_j = ctx.engine.multiply(params["w"][j], weight_decay)
        dL_dw_j = _ensure_level(ctx, ctx.engine.add(dL_dw[j], decay_j))
        new_w.append(_ensure_level(ctx, ctx.engine.subtract(params["w"][j], ctx.engine.multiply(dL_dw_j, lr_gate))))
    decay_b = ctx.engine.multiply(params["b"], weight_decay)
    dL_db_full = _ensure_level(ctx, ctx.engine.add(dL_db, decay_b))
    new_b = _ensure_level(ctx, ctx.engine.subtract(params["b"], ctx.engine.multiply(dL_db_full, lr_gate)))

    new_params = {
        "w": new_w,
        "b": new_b,
        "leaf_L": _ensure_level(ctx, ctx.engine.subtract(params["leaf_L"], ctx.engine.multiply(dL_dleaf_L, lr))),
        "leaf_R": _ensure_level(ctx, ctx.engine.subtract(params["leaf_R"], ctx.engine.multiply(dL_dleaf_R, lr))),
    }
    return new_params


def decrypt_params(ctx, params: dict, n_features: int, n_classes: int) -> dict:
    w = np.array([np.real(ctx.engine.decrypt(params["w"][j], ctx.sk))[0] for j in range(n_features)])
    b = np.real(ctx.engine.decrypt(params["b"], ctx.sk))[0]
    leaf_L = np.real(ctx.engine.decrypt(params["leaf_L"], ctx.sk))[:n_classes]
    leaf_R = np.real(ctx.engine.decrypt(params["leaf_R"], ctx.sk))[:n_classes]
    return {"w": w, "b": b, "leaf_L": leaf_L, "leaf_R": leaf_R, "steepness": STEEPNESS}


def main():
    """depth1_ckks.py의 main()과 같은 패턴 - 단일 프로세스로 몇 epoch만 빠르게 검증.
    (여러 epoch을 안전하게 돌리려면 depth1_ckks.py처럼 프로세스 격리 오케스트레이터가
    필요하다는 점도 동일 - 여기서는 '오블리크 gate가 CKKS에서 plaintext와 맞게 도는지'만
    확인하는 목적이라 2 epoch로 제한한다.)"""
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "iris"
    n_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
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
    params.pop("plaintext_init")

    print("[reference] running plaintext oblique trajectory for the same epoch counts...")
    ref_params_by_epoch = {}
    for k in range(n_epochs + 1):
        p = train_depthN_oblique(X_train, y_train_oh, depth=1, steepness=STEEPNESS, lr=lr, epochs=k, seed=seed)
        ref_params_by_epoch[k] = {
            "w": p["w"][0],
            "b": p["b"][0],
            "leaf_L": p["leaf_logits"][0],
            "leaf_R": p["leaf_logits"][1],
        }

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        params = forward_backward_update(ctx, dataset, params, sample_mask, n_features, n_classes, lr=lr)
        elapsed = time.time() - t0
        decoded = decrypt_params(ctx, params, n_features, n_classes)
        ref = ref_params_by_epoch[epoch]
        print(f"\n[epoch {epoch}] elapsed={elapsed:.1f}s")
        print(f"  w         ckks={np.round(decoded['w'],4)}  ref={np.round(ref['w'],4)}")
        print(f"  b         ckks={decoded['b']:.4f}  ref={ref['b']:.4f}")
        print(f"  leaf_L    ckks={np.round(decoded['leaf_L'],4)}  ref={np.round(ref['leaf_L'],4)}")
        print(f"  leaf_R    ckks={np.round(decoded['leaf_R'],4)}  ref={np.round(ref['leaf_R'],4)}")
        max_err = max(
            np.abs(decoded['w']-ref['w']).max(),
            abs(decoded['b']-ref['b']),
            np.abs(decoded['leaf_L']-ref['leaf_L']).max(),
            np.abs(decoded['leaf_R']-ref['leaf_R']).max(),
        )
        print(f"  max abs diff vs plaintext reference = {max_err:.5f}")
        gc.collect()


if __name__ == "__main__":
    main()
