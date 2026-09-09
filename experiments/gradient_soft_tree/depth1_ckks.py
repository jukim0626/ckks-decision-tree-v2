"""depth1_reference.py의 forward/backward/SGD update를 실제 CKKS 암호문 연산으로 그대로
옮긴 것 - MGI/Gini 계산이 없는 gradient soft tree를 처음으로 암호화 상태에서 검증하는
스크립트.

**설계 원칙**: alpha(feature attention)/threshold/leaf_logits는 전부 **private
training data로부터 학습되는 값**이므로 (closed_form_mgi의 threshold/soft-MGI weight와
같은 지위), 학습 내내 한 번도 decrypt하지 않고 ciphertext로만 갱신한다 - 이 프로젝트의
"client decrypt 없음" 원칙을 gradient descent 계보에도 그대로 적용.

**재사용한 기존 프리미티브**: `sigmoid_approx_enc`(ckks_tree.py), `ensure_level`/
`create_bootstrap_context`(closed_form_mgi/primitives.py), `scatter_to_slot`/
`next_power_of_two`(closed_form_mgi/simd_argmin.py), bounded Newton-Raphson reciprocal
패턴(soft_mgi.py의 z=D*y/w=target*y 트릭)을 MGI-무관한 일반 softmax에 재사용.

**새로 만든 것**: `packed_softmax()` - K개 값(패딩 포함 n_pow2 슬롯)의 일반 softmax
(exp(beta*x)/sum, MGI z-score 특화 로직 없음). exp 근사는 alpha/threshold/leaf_logits가
학습 중 실제로 도달하는 범위를 plaintext(`depth1_reference.py`)로 먼저 측정해서
(iris/wine/breast_cancer 전부 절댓값 2.0 미만) interval=[-2.5,2.5], beta=1.0으로 새로
피팅했다 (soft_mgi.py의 z-score용 beta=5는 이 용도로 쓰면 softmax가 너무 sharp해져서
attention 학습에 필요한 gradient가 죽는다 - 원래 sigmoid steepness를 annealing한 이유와
같은 문제라 재사용하지 않음).

**패딩 슬롯 처리**: threshold는 기존 closed_form_mgi 관례대로 전체 슬롯에 broadcast된
ciphertext라(=`ctx.engine.sum()`의 출력 형태와 동일), `sigmoid_approx_enc(enc_feature-
threshold)`가 padding 슬롯(n_samples..slot_count-1)에서도 0이 아닌 값을 만든다. 샘플
축으로 합산(`ctx.engine.sum`)하기 직전마다 `sample_mask`(1로 채워진 첫 n_samples 슬롯,
나머지 0)를 곱해서 이 오염을 제거한다 - 매 sum 호출 전에 한 번씩, soft_mgi.py의
valid_mask 재적용 원칙과 동일.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ckks_tree import sigmoid_approx_enc  # noqa: E402
from client_assisted.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from closed_form_mgi.exp_approx_coeffs import chebyshev_approximation_exp  # noqa: E402
from closed_form_mgi.primitives import create_bootstrap_context, ensure_level  # noqa: E402

# 2026-08-26: closed_form_mgi(이미 검증된 depth<=2 파이프라인)의 ensure_level 기본값
# min_level=8은 절대 안 건드림 - 대신 이 실험 파일 안에서만 낮은 값을 쓰는 로컬 래퍼를 둔다.
# depth=3 CKKS 학습이 CUDA OOM나는 근본 원인이 "파라미터의 level이 낮을수록 bootstrap이
# 훨씬 자주 걸리고, desilofhe는 프로세스 안에서 GPU 메모리를 절대 안 돌려주니 bootstrap
# 총 호출 횟수가 곧 peak 메모리"라는 걸 실측으로 확인함(EXPERIMENT_LOG 2026-08-26 참고).
# min_level을 낮추면 bootstrap 1번당 얻는 "여유 레벨"이 늘어나(26-min_level) 같은 그래프
# 깊이를 처리하는 데 필요한 총 bootstrap 횟수가 줄어든다. depthN_ckks.py의 최대
# un-checkpointed multiply 체인(threshold gradient 경로, 최대 3번 연속)에 여유를 두고
# 5로 설정 - 8보다 26-5=21 vs 26-8=18로 사이클당 여유가 약 17% 늘어난다.
_LOCAL_MIN_LEVEL = 5


def _ensure_level(ctx, ct):
    return ensure_level(ctx, ct, min_level=_LOCAL_MIN_LEVEL)
from closed_form_mgi.simd_argmin import next_power_of_two, scatter_to_slot  # noqa: E402
from closed_form_mgi.soft_mgi import _extract_weight_broadcast  # noqa: E402
from experiments.gradient_soft_tree.depth1_reference import (  # noqa: E402
    predict as plaintext_predict,
    softmax,
    softmax_backward,
    train_depth1 as plaintext_train_depth1,
)

STEEPNESS = 8.0  # SPLIT_SIGMOID_COEFFS(sigmoid_approx_coeffs.py)에 이미 baked-in
SOFTMAX_EXP_DEGREE = 20
SOFTMAX_EXP_INTERVAL = (-2.5, 2.5)
# 2026-08-26: soft_mgi.py의 z-score reciprocal은 z0(=D*y0)가 0.00025까지 작아질 수 있어서
# 22회가 필요했지만(2026-08-13 수렴 분석), 여기서는 z0=mean(rescaled exp_val)이 항상
# (0,1] 안이고 alpha/leaf_logits 값 자체가 크게 안 벌어지므로(plaintext 실측: 학습 내내
# |alpha|,|leaf_logits| < ~2) z0가 0.05 밑으로 내려가는 경우가 거의 없다. bounded NR은
# e_{k+1}=e_k^2 이차수렴이라, 최악 z0=0.05(e0=0.95)에서도 k=8이면 e_k~1e-4(CKKS 자체
# 노이즈 바닥과 비슷한 수준)로 이미 충분 - 10으로 안전마진을 조금 더 두고 씀. 22->10으로
# depth>=2에서 급증하는 노드당 softmax 호출 비용(=bootstrap 비용)을 줄이기 위한 조정
# (epoch당 소요시간이 depth마다 노드 수(2^depth-1)에 비례해서 커지므로, softmax 1회 비용을
# 낮추는 게 depth=3까지 실제로 돌릴 수 있느냐를 가르는 핵심 레버였다).
SOFTMAX_RECIP_ITERATIONS = 10


def softmax_exp_coeffs() -> list[float]:
    """exp(z)(z in [-2.5,2.5])의 Chebyshev 계수. exp(-beta*x) 피팅 함수에 beta=-1을 줘서
    exp(+x)를 얻는다 (chebyshev_approximation_exp는 beta 부호에 무관하게 동작)."""
    return chebyshev_approximation_exp(SOFTMAX_EXP_DEGREE, beta=-1.0, interval=SOFTMAX_EXP_INTERVAL).tolist()


_EXP_COEFFS = softmax_exp_coeffs()
_EXP_MAX = float(np.exp(SOFTMAX_EXP_INTERVAL[1]))  # 공개 상수: exp(2.5), rescale로 (0,1] 안에 가둠


def packed_softmax(ctx, values_packed, n_valid: int, n_pow2: int, reciprocal_iterations: int = SOFTMAX_RECIP_ITERATIONS):
    """values_packed(슬롯 0..n_valid-1에 값, 나머지 패딩 0) -> 합이 1인 softmax 벡터(같은 레이아웃).

    soft_mgi.py의 bounded Newton-Raphson 구조를 그대로 쓰되 z-score/MGI 특화 로직 없이
    raw value에 바로 적용 (exp_val을 공개 상한 exp(2.5)로 미리 나눠서 (0,1] 안에 가둬 두는
    것까지 동일한 트릭 - 이래야 z0=mean(exp_val)이 항상 <=1이라 bounded reciprocal이
    안전하게 수렴).
    """
    valid_mask = np.array([1.0] * n_valid + [0.0] * (n_pow2 - n_valid))
    z = ctx.engine.multiply(values_packed, valid_mask)
    z = ctx.engine.intt(z)
    # 2026-08-26: level_probe.py로 실측 - degree=20 exp poly가 레벨을 ~6 소모하고 뒤이은
    # 스칼라 곱 2번이 ~2 더 소모한다(합 ~8). 이전엔 여기 ensure_level이 없어서 values_packed가
    # (이전 epoch 갱신값이라) _LOCAL_MIN_LEVEL=5 근처에서 들어오면 poly 직후 레벨이 0 밑으로
    # 떨어져 "input ciphertext should have a positive level"로 죽었다(특히 level_preset=17
    # 테스트에서 epoch 2 재현). poly 진입 전에 미리 높은 min_level로 부트스트랩해서 방지.
    z = ensure_level(ctx, z, min_level=10)
    exp_val = ctx.engine.evaluate_polynomial(z, _EXP_COEFFS, ctx.rlk)
    exp_val = ctx.engine.multiply(exp_val, 1.0 / _EXP_MAX)
    exp_val = ctx.engine.multiply(exp_val, valid_mask)  # exp(0)=1 패딩 재오염 방지 (soft_mgi.py와 동일 버그 재발 방지)

    exp_val_for_sum = ctx.engine.intt(exp_val)
    denom = ctx.engine.sum(exp_val_for_sum, ctx.rotation_key)
    denom = _ensure_level(ctx, denom)
    exp_val = _ensure_level(ctx, exp_val)

    y0 = 1.0 / n_valid
    z_iter = ctx.engine.multiply(denom, y0)
    w = ctx.engine.multiply(exp_val, y0)
    for i in range(reciprocal_iterations):
        z_iter = _ensure_level(ctx, z_iter)
        w = _ensure_level(ctx, w)
        two_minus_z = ctx.engine.subtract(2.0, z_iter)
        z_new = ctx.engine.multiply(z_iter, two_minus_z, ctx.rlk)
        w = ctx.engine.multiply(w, two_minus_z, ctx.rlk)
        z_iter = z_new
        del two_minus_z
        if i % 5 == 0:
            gc.collect()
    return w


def softmax_backward_packed(ctx, dist_packed, dL_ddist_packed, n_valid: int, n_pow2: int):
    """packed dist/dL_ddist(둘 다 슬롯 0..n_valid-1)에 대해 dL/dlogit = dist*(dL_ddist - dot)
    (dot = sum_c dL_ddist_c*dist_c, 전체 슬롯에 broadcast). 패딩은 dist가 이미 0이라 자동으로 0."""
    dist_packed = _ensure_level(ctx, dist_packed)
    dL_ddist_packed = _ensure_level(ctx, dL_ddist_packed)
    prod = ctx.engine.multiply(dist_packed, dL_ddist_packed, ctx.rlk)
    prod = ctx.engine.intt(prod)
    dot = _ensure_level(ctx, ctx.engine.sum(prod, ctx.rotation_key))
    diff = _ensure_level(ctx, ctx.engine.subtract(dL_ddist_packed, dot))
    return ctx.engine.multiply(dist_packed, diff, ctx.rlk)


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
