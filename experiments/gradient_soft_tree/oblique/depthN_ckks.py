"""depthN_ckks.py(axis-aligned attention-blend)를 depth=1 oblique/depth1_ckks.py와
같은 방식으로 gate 부분만 바꿔서 임의 depth로 일반화한 것. leaf softmax/backward,
레벨별 g 전파 재귀 구조는 depthN_ckks.py와 100% 동일 - forward의 gate 계산과 backward의
"dL_dgate_i -> 파라미터 gradient" 변환 부분만 다르다.

파라미터: alpha/threshold(둘 다 feature별) -> w/b(선형결합 가중치+bias) 로 대체.
softmax(alpha)/packed_softmax 호출이 gate 쪽에서는 완전히 사라짐 (leaf 쪽 softmax는 유지)."""

from __future__ import annotations

import gc
import sys
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
STEEPNESS = 8.0

# 2026-09-08 추가 후 같은 날 폐기: fit interval을 [-2,2]->[-5,5]로 넓혀 재피팅해봤으나
# 결과가 애매해서(발산은 줄었지만 정확도/안정성이 뚜렷이 개선됐다고 보기 어려움) 원래
# [-2,2](axis-aligned baseline과 동일한 SPLIT_SIGMOID_COEFFS)로 되돌림. 다음 시도는
# 논문 기반 4단계 검증(활성화 함수/최적화/구조/포팅)으로 진행 - EXPERIMENT_LOG 참고.
# 코드는 참고용으로 남겨두되 현재 사용 안 함(_WIDE_SIGMOID_COEFFS를 쓰던 자리는 아래에서
# coeffs 인자를 생략해 axis-aligned와 동일한 기본값(SPLIT_SIGMOID_COEFFS)을 쓰도록 되돌림).
_WIDE_SIGMOID_COEFFS = chebyshev_approximation(degree=15, interval=5.0, steepness=STEEPNESS).tolist()

# 2026-09-08 논문 기반 검증 Phase 1 (TEL, arXiv 2002.07772 - 저차수 다항식이 구간 밖에서
# 훨씬 덜 폭발한다는 아이디어): interval은 baseline과 동일 [-2,2] 유지, degree만 15->5.
# numpy 사전분석: z=14에서 baseline(deg15) poly=-1.58e15인데 deg5는 poly=42277로
# 11자릿수 작음(구간 내 오차는 0.028->0.133로 커짐). CLAUDE.md의 "bootstrap 자체가 값
# 2~5 넘으면 조용히 깨진다"는 별개 제약이 있어 이것만으로 완전 해결은 안 될 수 있음 -
# 이 변수 하나만 바꿔서 "다항식 차수 문제"와 "bootstrap 크기 제약 문제"를 가른다.
_LOW_DEGREE_SIGMOID_COEFFS = chebyshev_approximation(degree=5, interval=2.0, steepness=STEEPNESS).tolist()


def _ensure_level(ctx, ct):
    return ensure_level(ctx, ct, min_level=_LOCAL_MIN_LEVEL)


def init_encrypted_params_N(ctx, n_features: int, n_classes: int, depth: int, seed: int, slot_count: int):
    """train_depthN_oblique과 완전히 같은 순서(w -> b -> leaf_logits)로 rng를 소비한다.
    2026-09-07 안정성 수정 #1: w 초기 표준편차를 1/sqrt(n_features)로 스케일링 - 이유는
    depthN_reference.py 상단 docstring 참고."""
    rng = np.random.default_rng(seed)
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth
    n_pow2_c = next_power_of_two(n_classes)

    w0 = rng.normal(0, 0.1 / np.sqrt(n_features), size=(n_internal, n_features))
    b0 = rng.normal(0, 0.1, size=(n_internal,))
    leaf0 = rng.normal(0, 0.1, size=(n_leaves, n_classes))

    w_cts = [
        [ctx.engine.encrypt([float(w0[i, j])] * slot_count, ctx.pk) for j in range(n_features)]
        for i in range(n_internal)
    ]
    b_cts = [ctx.engine.encrypt([float(b0[i])] * slot_count, ctx.pk) for i in range(n_internal)]
    leaf_cts = [
        ctx.engine.encrypt(leaf0[l].tolist() + [0.0] * (n_pow2_c - n_classes), ctx.pk) for l in range(n_leaves)
    ]
    return {"w": w_cts, "b": b_cts, "leaf_logits": leaf_cts}


def decrypt_params_N(ctx, params: dict, n_features: int, n_classes: int, depth: int) -> dict:
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth
    w = np.array(
        [[np.real(ctx.engine.decrypt(params["w"][i][j], ctx.sk))[0] for j in range(n_features)] for i in range(n_internal)]
    )
    b = np.array([np.real(ctx.engine.decrypt(params["b"][i], ctx.sk))[0] for i in range(n_internal)])
    leaf_logits = np.array(
        [np.real(ctx.engine.decrypt(params["leaf_logits"][l], ctx.sk))[:n_classes] for l in range(n_leaves)]
    )
    return {"w": w, "b": b, "leaf_logits": leaf_logits, "steepness": STEEPNESS, "depth": depth}


def forward_backward_update_N(
    ctx, dataset, params: dict, sample_mask, n_features: int, n_classes: int, depth: int, lr: float,
    weight_decay: float = 0.01,
):
    n_pow2_c = next_power_of_two(n_classes)
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth
    lr_gate = lr / np.sqrt(n_features)  # 안정성 수정 #2 - gate(w,b) 전용 lr, depthN_reference.py와 동일

    node_gate = [None] * n_internal
    node_parent_prob = [None] * n_internal

    # ---- forward: gate 계산만 다름 (선형결합 -> sigmoid 1번, softmax 없음) ----
    current_level_probs = [None]
    node_idx = 0
    for _level in range(depth):
        next_level_probs = []
        for parent_prob in current_level_probs:
            i = node_idx
            z = None
            for j in range(n_features):
                enc_feature = _ensure_level(ctx, dataset.enc_features[j])
                w_ij = _ensure_level(ctx, params["w"][i][j])
                term = ctx.engine.multiply(enc_feature, w_ij, ctx.rlk)
                z = term if z is None else ctx.engine.add(z, term)
            b_i = _ensure_level(ctx, params["b"][i])
            z = ctx.engine.add(z, b_i)
            # depthN_ckks.py(axis-aligned)와 동일한 이유: degree=15 sigmoid poly가 레벨을
            # ~5 소모하므로, poly 직후 레벨이 0 밑으로 안 떨어지게 진입 전 min_level=12로
            # 별도 보장한다 (그냥 _ensure_level(min_level=5)만으로는 부족함 - 2026-09-07
            # depth=2 epoch2에서 "should have a positive level" 실패로 확인).
            z = ensure_level(ctx, z, min_level=12)
            # 2026-09-08 Phase 1 검증: 저차수(degree=5) 게이트 다항식 단독 테스트.
            gate = sigmoid_approx_enc(ctx.engine, ctx.rlk, z, coeffs=_LOW_DEGREE_SIGMOID_COEFFS)

            node_gate[i], node_parent_prob[i] = gate, parent_prob

            if parent_prob is None:
                left = ctx.engine.subtract(1.0, gate)
                right = gate
            else:
                parent_prob = _ensure_level(ctx, parent_prob)
                left = ctx.engine.multiply(parent_prob, ctx.engine.subtract(1.0, gate), ctx.rlk)
                right = ctx.engine.multiply(parent_prob, gate, ctx.rlk)
            next_level_probs.append(_ensure_level(ctx, left))
            next_level_probs.append(_ensure_level(ctx, right))
            node_idx += 1
            gc.collect()
        current_level_probs = next_level_probs
    leaf_probs = current_level_probs

    # ---- leaf/backward 상단부: depthN_ckks.py와 100% 동일 ----
    leafdist = [packed_softmax(ctx, params["leaf_logits"][l], n_classes, n_pow2_c) for l in range(n_leaves)]

    y_hat = []
    for c in range(n_classes):
        acc = None
        for l in range(n_leaves):
            ld_c = _ensure_level(ctx, _extract_weight_broadcast(ctx, leafdist[l], c))
            term = ctx.engine.multiply(leaf_probs[l], ld_c, ctx.rlk)
            acc = term if acc is None else ctx.engine.add(acc, term)
        y_hat.append(_ensure_level(ctx, acc))

    n_samples = dataset.n_samples
    dL_dyhat = []
    for c in range(n_classes):
        diff = ctx.engine.subtract(y_hat[c], dataset.enc_labels[c])
        dL_dyhat.append(_ensure_level(ctx, ctx.engine.multiply(diff, 2.0 / n_samples)))

    current_g = [None] * n_leaves
    dL_dleaf_logits = [None] * n_leaves
    for l in range(n_leaves):
        dL_ddist_packed = None
        g_l = None
        for c in range(n_classes):
            term = ctx.engine.multiply(dL_dyhat[c], leaf_probs[l], ctx.rlk)
            term = ctx.engine.multiply(term, sample_mask, ctx.rlk)
            term = ctx.engine.intt(term)
            s = _ensure_level(ctx, ctx.engine.sum(term, ctx.rotation_key))
            piece = scatter_to_slot(ctx, s, c)
            dL_ddist_packed = piece if dL_ddist_packed is None else ctx.engine.add(dL_ddist_packed, piece)

            ld_c = _extract_weight_broadcast(ctx, leafdist[l], c)
            g_piece = ctx.engine.multiply(dL_dyhat[c], ld_c, ctx.rlk)
            g_l = g_piece if g_l is None else ctx.engine.add(g_l, g_piece)
        dL_dleaf_logits[l] = softmax_backward_packed(ctx, leafdist[l], _ensure_level(ctx, dL_ddist_packed), n_classes, n_pow2_c)
        current_g[l] = _ensure_level(ctx, g_l)
        gc.collect()

    # ---- 레벨을 거슬러 올라가며 노드 gradient 계산 - 여기 gate backward 부분만 다름 ----
    new_w = [None] * n_internal
    new_b = [None] * n_internal
    for level in reversed(range(depth)):
        start = (1 << level) - 1
        count = 1 << level
        next_g = [None] * count
        for idx in range(count):
            i = start + idx
            g_left, g_right = current_g[2 * idx], current_g[2 * idx + 1]
            gate, p_i = node_gate[i], node_parent_prob[i]

            diff_g = _ensure_level(ctx, ctx.engine.subtract(g_right, g_left))
            if p_i is None:
                dL_dgate_i = diff_g
            else:
                dL_dgate_i = _ensure_level(ctx, ctx.engine.multiply(p_i, diff_g, ctx.rlk))

            if i != 0:
                one_minus_gate = ctx.engine.subtract(1.0, gate)
                term_l = ctx.engine.multiply(g_left, one_minus_gate, ctx.rlk)
                term_r = ctx.engine.multiply(g_right, gate, ctx.rlk)
                next_g[idx] = _ensure_level(ctx, ctx.engine.add(term_l, term_r))

            # --- 오블리크 gate backward: dL_dgate_i -> dL_dz -> dL_dw/dL_db (softmax_backward 불필요) ---
            surrogate = _ensure_level(ctx, ctx.engine.multiply(gate, ctx.engine.subtract(1.0, gate), ctx.rlk))
            dL_dz = ctx.engine.multiply(dL_dgate_i, surrogate, ctx.rlk)
            dL_dz = _ensure_level(ctx, ctx.engine.multiply(dL_dz, STEEPNESS))

            new_w_i = []
            for j in range(n_features):
                enc_feature = _ensure_level(ctx, dataset.enc_features[j])
                prod = ctx.engine.multiply(dL_dz, enc_feature, ctx.rlk)
                prod = ctx.engine.multiply(prod, sample_mask, ctx.rlk)
                prod = ctx.engine.intt(prod)
                sum_j = _ensure_level(ctx, ctx.engine.sum(prod, ctx.rotation_key))
                # 안정성 수정 #3: L2 weight decay (dL_dw += weight_decay*w), #2: lr_gate 사용
                decay_j = ctx.engine.multiply(params["w"][i][j], weight_decay)
                dL_dw_ij = _ensure_level(ctx, ctx.engine.add(sum_j, decay_j))
                new_w_i.append(
                    _ensure_level(ctx, ctx.engine.subtract(params["w"][i][j], ctx.engine.multiply(dL_dw_ij, lr_gate)))
                )
                gc.collect()
            new_w[i] = new_w_i

            prod_b = ctx.engine.multiply(dL_dz, sample_mask, ctx.rlk)
            prod_b = ctx.engine.intt(prod_b)
            sum_b = _ensure_level(ctx, ctx.engine.sum(prod_b, ctx.rotation_key))
            decay_b = ctx.engine.multiply(params["b"][i], weight_decay)
            dL_db_i = _ensure_level(ctx, ctx.engine.add(sum_b, decay_b))
            new_b[i] = _ensure_level(ctx, ctx.engine.subtract(params["b"][i], ctx.engine.multiply(dL_db_i, lr_gate)))
        current_g = next_g
        gc.collect()

    new_leaf_logits = [
        _ensure_level(ctx, ctx.engine.subtract(params["leaf_logits"][l], ctx.engine.multiply(dL_dleaf_logits[l], lr)))
        for l in range(n_leaves)
    ]

    return {"w": new_w, "b": new_b, "leaf_logits": new_leaf_logits}


def _debug_validate(dataset_name: str, depth: int, n_epochs: int, lr: float, seed: int = 0, level_preset: int | None = None):
    import time

    X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name)
    n_features = X_train.shape[1]
    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    y_train_oh = one_hot_encode(y_train, n_classes)

    print(f"[setup] dataset={dataset_name} depth={depth} n_features={n_features} n_classes={n_classes} level_preset={level_preset}")
    ctx = create_bootstrap_context(mode="gpu", level_preset=level_preset)
    dataset = encrypt_dataset(ctx, X_train, y_train_oh)
    sample_mask = ctx.engine.encrypt([1.0] * dataset.n_samples, ctx.pk)

    params = init_encrypted_params_N(ctx, n_features, n_classes, depth, seed=seed, slot_count=ctx.engine.slot_count)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        params = forward_backward_update_N(ctx, dataset, params, sample_mask, n_features, n_classes, depth, lr=lr)
        elapsed = time.time() - t0
        decoded = decrypt_params_N(ctx, params, n_features, n_classes, depth)
        ref = train_depthN_oblique(X_train, y_train_oh, depth=depth, lr=lr, epochs=epoch, seed=seed)
        max_err = max(
            np.abs(decoded["w"] - ref["w"]).max(),
            np.abs(decoded["b"] - ref["b"]).max(),
            np.abs(decoded["leaf_logits"] - ref["leaf_logits"]).max(),
        )
        print(f"[epoch {epoch}] elapsed={elapsed:.1f}s  max abs diff vs plaintext = {max_err:.5f}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "iris"
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    lr_ = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    lp = int(sys.argv[5]) if len(sys.argv) > 5 else None
    _debug_validate(ds, d, n_ep, lr_, level_preset=lp)
