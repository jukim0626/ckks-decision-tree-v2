"""depthN_ckks.forward_backward_update_N을 TreeConfig로 각 최적화를 독립적으로 켜고 끌 수
있게 재구성한 버전. baseline(experiments/gradient_soft_tree/depthN_ckks.py)은 절대
import/수정하지 않는다 - 이 파일은 그 파일의 로직을 복제한 뒤 분기만 추가했다.

TreeConfig() 기본값으로 호출하면 baseline과 수치적으로 동일해야 한다 (opt/validate.py로
검증). init_encrypted_params_N/decrypt_params_N은 baseline 것을 그대로 재사용한다 - encoding
자체(alpha/threshold/leaf_logits ciphertext 레이아웃)는 모든 Phase에서 안 바뀌기 때문
(Phase 2/3은 "그 ciphertext를 softmax 없이 어떻게 해석하느냐"만 바꾼다)."""

from __future__ import annotations

import gc

import numpy as np

from core.approximation.sigmoid import sigmoid_approx_enc
from core.encrypted_ops.slot_packing import next_power_of_two, scatter_to_slot
from core.encrypted_ops.slot_packing import extract_weight_broadcast as _extract_weight_broadcast
from core.encrypted_ops.softmax import softmax_exp_coeffs, _EXP_MAX
from models.gradient_soft_tree.opt.config import TreeConfig
from models.gradient_soft_tree.opt.poly_gate import (
    gate_coeffs,
    gate_derivative_coeffs,
    gate_entry_min_level,
)
from models.gradient_soft_tree.opt.profiler import batch_ensure_level_profiled, ensure_level_profiled

_EXP_COEFFS = softmax_exp_coeffs()


def _lvl(ctx, ct, tag, min_level, profiler):
    return ensure_level_profiled(ctx, ct, tag, min_level, profiler)


def resolve_gate_entry_min_level(config: TreeConfig) -> int:
    if config.gate_entry_min_level is not None:
        return config.gate_entry_min_level
    return gate_entry_min_level(config.gate_degree)


def packed_softmax_opt(ctx, values_packed, n_valid: int, n_pow2: int, config: TreeConfig, profiler, tag: str):
    """depth1_ckks.packed_softmax와 동일 로직, reciprocal_iterations만 config에서 읽고
    모든 ensure_level 호출에 tag를 붙인다."""
    valid_mask = np.array([1.0] * n_valid + [0.0] * (n_pow2 - n_valid))
    z = ctx.engine.multiply(values_packed, valid_mask)
    z = ctx.engine.intt(z)
    z = _lvl(ctx, z, tag, config.softmax_entry_min_level, profiler)
    exp_val = ctx.engine.evaluate_polynomial(z, _EXP_COEFFS, ctx.rlk)
    exp_val = ctx.engine.multiply(exp_val, 1.0 / _EXP_MAX)
    exp_val = ctx.engine.multiply(exp_val, valid_mask)

    exp_val_for_sum = ctx.engine.intt(exp_val)
    denom = ctx.engine.sum(exp_val_for_sum, ctx.rotation_key)
    denom = _lvl(ctx, denom, tag, config.local_min_level, profiler)
    exp_val = _lvl(ctx, exp_val, tag, config.local_min_level, profiler)

    y0 = 1.0 / n_valid
    z_iter = ctx.engine.multiply(denom, y0)
    w = ctx.engine.multiply(exp_val, y0)
    for i in range(config.reciprocal_iterations):
        z_iter = _lvl(ctx, z_iter, tag, config.local_min_level, profiler)
        w = _lvl(ctx, w, tag, config.local_min_level, profiler)
        two_minus_z = ctx.engine.subtract(2.0, z_iter)
        z_new = ctx.engine.multiply(z_iter, two_minus_z, ctx.rlk)
        w = ctx.engine.multiply(w, two_minus_z, ctx.rlk)
        z_iter = z_new
        del two_minus_z
        if i % 5 == 0:
            gc.collect()
    return w


def softmax_backward_packed_opt(ctx, dist_packed, dL_ddist_packed, n_valid: int, n_pow2: int, config, profiler, tag: str):
    dist_packed = _lvl(ctx, dist_packed, tag, config.local_min_level, profiler)
    dL_ddist_packed = _lvl(ctx, dL_ddist_packed, tag, config.local_min_level, profiler)
    prod = ctx.engine.multiply(dist_packed, dL_ddist_packed, ctx.rlk)
    prod = ctx.engine.intt(prod)
    dot = _lvl(ctx, ctx.engine.sum(prod, ctx.rotation_key), tag, config.local_min_level, profiler)
    diff = _lvl(ctx, ctx.engine.subtract(dL_ddist_packed, dot), tag, config.local_min_level, profiler)
    return ctx.engine.multiply(dist_packed, diff, ctx.rlk)


def attention_regularizer_grad(ctx, alpha_ct, n_features: int, n_pow2_f: int, config: TreeConfig, profiler, tag: str):
    """Phase 3: dR/da_ij for R = lambda_sum*(sum_j a_ij - 1)^2 + lambda_binary*sum_j a_ij^2(1-a_ij)^2.
    division/exp/softmax/comparison 없이 +,-,x 만 사용 (문서 Phase 3 제약)."""
    # 2026-08-27 실측으로 발견한 버그 수정: alpha_ct는 n_features(<=n_pow2_f)개 실제
    # 값만 채워서 encrypt한 packed ciphertext라, 나머지 슬롯(n_features..slot_count-1)이
    # 항상 0이라는 보장이 없다(이 파일 다른 모든 packed 연산 - packed_softmax의
    # valid_mask, scatter_to_slot의 slot-0 mask 등 - 이 전부 sum 전에 반드시 마스킹하는
    # 이유). 여기만 마스킹 없이 ctx.engine.sum()을 호출해서 (1) 패딩 슬롯의 쓰레기값까지
    # 합산에 끼어들고, (2) engine.sum()이 결과를 전체 슬롯에 broadcast하는 관례상 grad
    # 자체도 마스킹 없이 패딩 슬롯에 그대로 쓰여, 다음 epoch에 패딩이 오염된 채로 다시
    # sum()에 들어가 값이 기하급수적으로 폭주했다(20-epoch 검증 실행에서 max_abs_diff가
    # 2.1e83까지 발산 - opt/inspect_param_magnitudes.py로 alpha만 유독 4개 feature가
    # 거의 동일하게 이동한 패턴을 보고 역추적함). valid_mask를 sum 전/grad 적용 후 양쪽에
    # 곱해서 패딩 슬롯을 항상 0으로 유지한다.
    valid_mask = np.array([1.0] * n_features + [0.0] * (n_pow2_f - n_features))
    grad = None
    if config.lambda_sum != 0.0:
        s = ctx.engine.multiply(alpha_ct, valid_mask)
        s = ctx.engine.intt(s)
        s = ctx.engine.sum(s, ctx.rotation_key)
        s = _lvl(ctx, s, tag, config.local_min_level, profiler)
        term = ctx.engine.multiply(ctx.engine.subtract(s, 1.0), 2.0 * config.lambda_sum)
        term = ctx.engine.multiply(term, valid_mask)
        grad = term
    if config.lambda_binary != 0.0:
        a = ctx.engine.multiply(alpha_ct, valid_mask)
        a = _lvl(ctx, a, tag, config.local_min_level, profiler)
        one_minus_a = ctx.engine.subtract(1.0, a)
        one_minus_2a = ctx.engine.subtract(1.0, ctx.engine.multiply(a, 2.0))
        t1 = ctx.engine.multiply(a, one_minus_a, ctx.rlk)
        t2 = ctx.engine.multiply(t1, one_minus_2a, ctx.rlk)
        term = ctx.engine.multiply(t2, 2.0 * config.lambda_binary)
        term = ctx.engine.multiply(term, valid_mask)
        grad = term if grad is None else ctx.engine.add(grad, term)
    return grad


def gate_forward(ctx, enc_diff, config: TreeConfig, profiler, tag: str):
    """gate_j = P(x-theta). config.gate_degree==15면 baseline sigmoid poly와 동일 계수."""
    enc_diff = _lvl(ctx, enc_diff, tag, resolve_gate_entry_min_level(config), profiler)
    if config.gate_degree == 15:
        gate_j = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff)
    else:
        gate_j = ctx.engine.evaluate_polynomial(enc_diff, gate_coeffs(config.gate_degree), ctx.rlk)
    return enc_diff, gate_j


def gate_surrogate(ctx, config: TreeConfig, z_j, gate_j, profiler, tag: str):
    """backward에서 쓰는 P'(z) 근사. use_true_gate_derivative=False면 baseline과 똑같이
    gate_j*(1-gate_j) 로지스틱 근사(=degree 무관 baseline 재현용). True면 실제 fit된
    다항식의 정확한 도함수(P'(z))를 z_j에 직접 평가 - Phase 4 conceptual change("P(x-theta)
    자체가 모델")에 맞는 엄밀한 gradient."""
    if not config.use_true_gate_derivative:
        return ctx.engine.multiply(gate_j, ctx.engine.subtract(1.0, gate_j), ctx.rlk)
    z_j = _lvl(ctx, z_j, tag, config.local_min_level, profiler)
    return ctx.engine.evaluate_polynomial(z_j, gate_derivative_coeffs(config.gate_degree), ctx.rlk)


def forward_backward_update_N_opt(
    ctx, dataset, params: dict, sample_mask, n_features: int, n_classes: int, depth: int, lr: float,
    config: TreeConfig, profiler=None,
):
    n_pow2_f = next_power_of_two(n_features)
    n_pow2_c = next_power_of_two(n_classes)
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth

    node_gate = [None] * n_internal
    node_gate_j = [None] * n_internal
    node_z_j = [None] * n_internal
    node_w = [None] * n_internal
    node_parent_prob = [None] * n_internal

    current_level_probs = [None]
    node_idx = 0
    for _level in range(depth):
        next_level_probs = []
        for parent_prob in current_level_probs:
            i = node_idx
            if config.attention_softmax:
                w = packed_softmax_opt(ctx, params["alpha"][i], n_features, n_pow2_f, config, profiler, "attention_softmax")
            else:
                w = params["alpha"][i]  # raw a_ij, softmax 없음 (Phase 3)

            gate_terms = []
            z_terms = []
            gate = None
            for j in range(n_features):
                enc_feature = _lvl(ctx, dataset.enc_features[j], "routing_forward", config.local_min_level, profiler)
                threshold_j = _lvl(ctx, params["threshold"][i][j], "routing_forward", config.local_min_level, profiler)
                enc_diff = ctx.engine.subtract(enc_feature, threshold_j)
                z_j, gate_j = gate_forward(ctx, enc_diff, config, profiler, "sigmoid_forward")
                gate_terms.append(gate_j)
                z_terms.append(z_j)
                w_j = _extract_weight_broadcast(ctx, w, j)
                piece = ctx.engine.multiply(w_j, gate_j, ctx.rlk)
                gate = piece if gate is None else ctx.engine.add(gate, piece)
            gate = _lvl(ctx, gate, "routing_forward", config.local_min_level, profiler)

            node_gate[i], node_gate_j[i], node_z_j[i], node_w[i], node_parent_prob[i] = (
                gate, gate_terms, z_terms, w, parent_prob
            )

            if parent_prob is None:
                left = ctx.engine.subtract(1.0, gate)
                right = gate
            else:
                parent_prob = _lvl(ctx, parent_prob, "routing_forward", config.local_min_level, profiler)
                left = ctx.engine.multiply(parent_prob, ctx.engine.subtract(1.0, gate), ctx.rlk)
                right = ctx.engine.multiply(parent_prob, gate, ctx.rlk)
            next_level_probs.append(_lvl(ctx, left, "routing_forward", config.local_min_level, profiler))
            next_level_probs.append(_lvl(ctx, right, "routing_forward", config.local_min_level, profiler))
            node_idx += 1
            gc.collect()
        current_level_probs = next_level_probs
    leaf_probs = current_level_probs

    if config.leaf_softmax:
        leafdist = [
            packed_softmax_opt(ctx, params["leaf_logits"][l], n_classes, n_pow2_c, config, profiler, "leaf_softmax")
            for l in range(n_leaves)
        ]
    else:
        leafdist = [params["leaf_logits"][l] for l in range(n_leaves)]  # 직접 leaf score (Phase 2)

    y_hat = []
    for c in range(n_classes):
        acc = None
        for l in range(n_leaves):
            ld_c = _lvl(ctx, _extract_weight_broadcast(ctx, leafdist[l], c), "routing_forward", config.local_min_level, profiler)
            term = ctx.engine.multiply(leaf_probs[l], ld_c, ctx.rlk)
            acc = term if acc is None else ctx.engine.add(acc, term)
        y_hat.append(_lvl(ctx, acc, "routing_forward", config.local_min_level, profiler))

    n_samples = dataset.n_samples
    dL_dyhat = []
    for c in range(n_classes):
        diff = ctx.engine.subtract(y_hat[c], dataset.enc_labels[c])
        dL_dyhat.append(_lvl(ctx, ctx.engine.multiply(diff, 2.0 / n_samples), "loss", config.local_min_level, profiler))

    current_g = [None] * n_leaves
    dL_dleaf_logits = [None] * n_leaves
    for l in range(n_leaves):
        # Phase 5A 확장: backward_leaf도 (leaf, class) 쌍마다 개별 reactive guard였던 걸
        # (Phase 0/wine 실측 둘 다 n_leaves*n_classes에서 거의 100% 트리거) threshold
        # backward와 같은 2-pass 패턴으로 바꾼다 - 클래스별 sum을 전부 미리(guard 없이)
        # 계산해두고 한 번에(hoist_backward=True면 merge_bootstrap 페어링) 새로 고친다.
        s_raw_list = [None] * n_classes
        g_piece_list = [None] * n_classes
        for c in range(n_classes):
            term = ctx.engine.multiply(dL_dyhat[c], leaf_probs[l], ctx.rlk)
            term = ctx.engine.multiply(term, sample_mask, ctx.rlk)
            term = ctx.engine.intt(term)
            s_raw_list[c] = ctx.engine.sum(term, ctx.rotation_key)  # guard는 아래 batch 단계에서

            ld_c = _extract_weight_broadcast(ctx, leafdist[l], c)
            g_piece_list[c] = ctx.engine.multiply(dL_dyhat[c], ld_c, ctx.rlk)

        if config.hoist_backward:
            s_fresh = batch_ensure_level_profiled(
                ctx, [(s_raw_list[c], "backward_leaf") for c in range(n_classes)],
                config.local_min_level, profiler,
            )
        else:
            s_fresh = [
                _lvl(ctx, s_raw_list[c], "backward_leaf", config.local_min_level, profiler)
                for c in range(n_classes)
            ]

        dL_ddist_packed = None
        g_l = None
        for c in range(n_classes):
            piece = scatter_to_slot(ctx, s_fresh[c], c)
            dL_ddist_packed = piece if dL_ddist_packed is None else ctx.engine.add(dL_ddist_packed, piece)
            g_l = g_piece_list[c] if g_l is None else ctx.engine.add(g_l, g_piece_list[c])

        if config.leaf_softmax:
            dL_dleaf_logits[l] = softmax_backward_packed_opt(
                ctx, leafdist[l], _lvl(ctx, dL_ddist_packed, "backward_leaf", config.local_min_level, profiler),
                n_classes, n_pow2_c, config, profiler, "backward_leaf",
            )
        else:
            # y_hat_c = sum_l leaf_prob_l * v_{l,c} (직접 score) -> dL/dv_{l,c} = dL_ddist_packed 그대로
            dL_dleaf_logits[l] = _lvl(ctx, dL_ddist_packed, "backward_leaf", config.local_min_level, profiler)
            if config.lambda_leaf != 0.0:
                reg_grad = ctx.engine.multiply(params["leaf_logits"][l], 2.0 * config.lambda_leaf)
                dL_dleaf_logits[l] = ctx.engine.add(dL_dleaf_logits[l], reg_grad)
        current_g[l] = _lvl(ctx, g_l, "backward_leaf", config.local_min_level, profiler)
        gc.collect()

    new_threshold = [None] * n_internal
    new_alpha = [None] * n_internal
    for level in reversed(range(depth)):
        start = (1 << level) - 1
        count = 1 << level
        next_g = [None] * count
        for idx in range(count):
            i = start + idx
            g_left, g_right = current_g[2 * idx], current_g[2 * idx + 1]
            gate, w, gate_terms, z_terms, p_i = node_gate[i], node_w[i], node_gate_j[i], node_z_j[i], node_parent_prob[i]

            diff_g = _lvl(ctx, ctx.engine.subtract(g_right, g_left), "backward_gate", config.local_min_level, profiler)
            if p_i is None:
                dL_dgate_i = diff_g
            else:
                dL_dgate_i = _lvl(ctx, ctx.engine.multiply(p_i, diff_g, ctx.rlk), "backward_gate", config.local_min_level, profiler)

            if i != 0:
                one_minus_gate = ctx.engine.subtract(1.0, gate)
                term_l = ctx.engine.multiply(g_left, one_minus_gate, ctx.rlk)
                term_r = ctx.engine.multiply(g_right, gate, ctx.rlk)
                next_g[idx] = _lvl(ctx, ctx.engine.add(term_l, term_r), "backward_gate", config.local_min_level, profiler)

            # Phase 5A (2026-08-27 재설계): 처음엔 gate_terms/w_j를 loop 진입 전에 hoist
            # 했었는데, opt/profiler.py의 checks_summary 실측(20-epoch 검증 실행, epoch
            # 13/15)으로 확인해보니 gate_j/surrogate/sum_t/w_j는 trigger_rate가 항상 0%에
            # 가깝고, 실제 병목은 매 feature 체인의 **마지막** 단계
            # (backward_threshold_dLdtj, 43~57% 트리거)였다 - 앞쪽을 hoist하는 건 애초에
            # 병목이 아닌 곳을 최적화하는 것이었다. 그래서 계산을 두 pass로 나눠 "체인 끝의
            # 곱셈 직전 값"까지는 guard 없이 4-feature 전부 계산해두고, 그 4개를 한 번에(
            # hoist_backward=True면 merge_bootstrap 페어링, False면 기존과 동일한 개별
            # reactive guard) 새로 고친 뒤 마지막 곱셈을 마무리한다. attention 쪽의 동등한
            # 마지막 단계(backward_attention_suma)도 같은 패턴으로 묶는다 - baseline(softmax
            # 있음)에서는 이쪽도 Phase 0 실측상 28/28(100%) 트리거였다.
            dL_dalpha_packed = None
            new_thresh_i = [None] * n_features
            sum_t_list = [None] * n_features
            mix_list = [None] * n_features
            sum_a_raw_list = [None] * n_features
            gate_j_list = [None] * n_features

            for j in range(n_features):
                gate_j = _lvl(ctx, gate_terms[j], "backward_threshold_gatej", config.local_min_level, profiler)
                gate_j_list[j] = gate_j
                surrogate = gate_surrogate(ctx, config, z_terms[j], gate_j, profiler, "backward_threshold_surrogate")
                surrogate = _lvl(ctx, surrogate, "backward_threshold_surrogate", config.local_min_level, profiler)

                # softmax든 raw a_ij든 threshold_j gradient는 항상 이 슬롯 값을 곱한다
                # (softmax=True면 w_j=softmax(alpha)_j, False면 a_ij 그 자체 - 두 경우가
                # 원래 코드에서 별개 변수(w_j/a_j)였지만 동일한 extract+multiply라 통합).
                mix_list[j] = _lvl(ctx, _extract_weight_broadcast(ctx, w, j), "backward_attention_wj", config.local_min_level, profiler)

                prod_t = ctx.engine.multiply(dL_dgate_i, surrogate, ctx.rlk)
                prod_t = ctx.engine.multiply(prod_t, sample_mask, ctx.rlk)
                prod_t = ctx.engine.intt(prod_t)
                sum_t_list[j] = ctx.engine.sum(prod_t, ctx.rotation_key)  # guard는 아래 batch 단계에서

                if config.attention_softmax:
                    # softmax 재매개변수화의 closed-form: dL/dalpha_j = w_j * sum_samples(dL_dgate*(gate_j-gate))
                    term_a = ctx.engine.multiply(dL_dgate_i, ctx.engine.subtract(gate_j, gate), ctx.rlk)
                else:
                    # 직접 파라미터: dL/da_ij = sum_samples(dL_dgate * gate_j) (softmax jacobian 없음, Phase 3)
                    term_a = ctx.engine.multiply(dL_dgate_i, gate_j, ctx.rlk)
                term_a = ctx.engine.multiply(term_a, sample_mask, ctx.rlk)
                term_a = ctx.engine.intt(term_a)
                sum_a_raw_list[j] = ctx.engine.sum(term_a, ctx.rotation_key)  # guard는 아래 batch 단계에서
                gc.collect()

            if config.hoist_backward:
                sum_t_fresh = batch_ensure_level_profiled(
                    ctx, [(sum_t_list[j], "backward_threshold_sumt") for j in range(n_features)],
                    config.local_min_level, profiler,
                )
                sum_a_fresh = batch_ensure_level_profiled(
                    ctx, [(sum_a_raw_list[j], "backward_attention_suma") for j in range(n_features)],
                    config.local_min_level, profiler,
                )
            else:
                sum_t_fresh = [
                    _lvl(ctx, sum_t_list[j], "backward_threshold_sumt", config.local_min_level, profiler)
                    for j in range(n_features)
                ]
                sum_a_fresh = [
                    _lvl(ctx, sum_a_raw_list[j], "backward_attention_suma", config.local_min_level, profiler)
                    for j in range(n_features)
                ]

            for j in range(n_features):
                dL_dt_j = _lvl(
                    ctx, ctx.engine.multiply(sum_t_fresh[j], mix_list[j], ctx.rlk),
                    "backward_threshold_dLdtj", config.local_min_level, profiler,
                )
                # use_true_gate_derivative=True면 surrogate가 이미 P'(diff) 그 자체라
                # d(diff)/d(theta)=-1만 곱한다 (opt/grad_check.py로 실측 검증된 버그 수정 -
                # steepness=8은 gate_j*(1-gate_j) 로지스틱 surrogate 전용 상수라 P'(diff)에
                # 또 곱하면 안 된다).
                theta_chain_const = -1.0 if config.use_true_gate_derivative else -8.0
                dL_dt_j = ctx.engine.multiply(dL_dt_j, theta_chain_const)
                new_thresh_i[j] = _lvl(
                    ctx, ctx.engine.subtract(params["threshold"][i][j], ctx.engine.multiply(dL_dt_j, lr)),
                    "parameter_update", config.local_min_level, profiler,
                )

                piece_a = scatter_to_slot(ctx, sum_a_fresh[j], j)
                dL_dalpha_packed = piece_a if dL_dalpha_packed is None else ctx.engine.add(dL_dalpha_packed, piece_a)

            dL_dalpha_packed = _lvl(ctx, dL_dalpha_packed, "backward_attention_combine", config.local_min_level, profiler)
            if config.attention_softmax:
                w_lvl = _lvl(ctx, w, "backward_attention_combine", config.local_min_level, profiler)
                dL_dalpha_i = ctx.engine.multiply(w_lvl, dL_dalpha_packed, ctx.rlk)
            else:
                dL_dalpha_i = dL_dalpha_packed
                reg_grad = attention_regularizer_grad(ctx, params["alpha"][i], n_features, n_pow2_f, config, profiler, "backward_attention_combine")
                if reg_grad is not None:
                    reg_grad = _lvl(ctx, reg_grad, "backward_attention_combine", config.local_min_level, profiler)
                    dL_dalpha_i = ctx.engine.add(dL_dalpha_i, reg_grad)
            new_alpha[i] = _lvl(
                ctx, ctx.engine.subtract(params["alpha"][i], ctx.engine.multiply(dL_dalpha_i, lr)),
                "parameter_update", config.local_min_level, profiler,
            )
            new_threshold[i] = new_thresh_i
        current_g = next_g
        gc.collect()

    new_leaf_logits = [
        _lvl(ctx, ctx.engine.subtract(params["leaf_logits"][l], ctx.engine.multiply(dL_dleaf_logits[l], lr)),
             "parameter_update", config.local_min_level, profiler)
        for l in range(n_leaves)
    ]

    return {"alpha": new_alpha, "threshold": new_threshold, "leaf_logits": new_leaf_logits}
