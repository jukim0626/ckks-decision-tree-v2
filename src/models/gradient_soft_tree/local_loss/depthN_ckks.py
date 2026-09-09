"""local_loss/depthN_reference.py(plaintext, ReBoot local-loss block 아이디어)의 CKKS
이식판. baseline `depthN_ckks.py`는 절대 안 건드림 - forward(gate 계산)는 baseline과
100% 동일한 코드를 그대로 재사용하고, **레벨마다 local classifier로 바로 loss를 계산해서
그 레벨 자신의 alpha/threshold만 업데이트**하는 학습 알고리즘만 다르다(depthN_ckks.py
docstring 및 local_loss/depthN_reference.py 상단 설명 참고).

핵심 기대 효과: baseline은 backward가 leaf(가장 깊은 레벨)에서 root까지 전부 거슬러
올라가야 해서 곱셈 depth가 tree depth에 비례했다. 이 버전은 각 레벨의 backward가 그
레벨 안에서 끝나므로(다음 레벨로 gradient가 안 넘어감), **레벨 하나를 처리하는 데
필요한 곱셈 depth가 tree depth와 무관하게 상수**여야 한다(ReBoot 논문 Table 1과 같은
구조)."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.ckks_engine import create_bootstrap_context, ensure_level  # noqa: E402
from core.data.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from core.encrypted_ops.slot_packing import extract_weight_broadcast, next_power_of_two, scatter_to_slot  # noqa: E402
from core.approximation.sigmoid import STEEPNESS  # noqa: E402
from core.encrypted_ops.softmax import packed_softmax, softmax_backward_packed  # noqa: E402
from models.gradient_soft_tree.gate import compute_axis_aligned_gate  # noqa: E402
from models.gradient_soft_tree.local_loss.depthN_reference import train_depthN_local_loss  # noqa: E402

_LOCAL_MIN_LEVEL = 5


def _ensure_level(ctx, ct):
    return ensure_level(ctx, ct, min_level=_LOCAL_MIN_LEVEL)


def init_encrypted_params_N(ctx, n_features: int, n_classes: int, depth: int, seed: int, slot_count: int):
    """local_loss/depthN_reference.train_depthN_local_loss와 완전히 같은 순서로 rng를
    소비한다(alpha -> threshold -> local_logits[0] -> local_logits[1] -> ...)."""
    rng = np.random.default_rng(seed)
    n_internal = (1 << depth) - 1
    n_pow2_f = next_power_of_two(n_features)
    n_pow2_c = next_power_of_two(n_classes)

    alpha0 = rng.normal(0, 0.1, size=(n_internal, n_features))
    threshold0 = rng.normal(0, 0.1, size=(n_internal, n_features))
    local_logits0 = [rng.normal(0, 0.1, size=(1 << (lvl + 1), n_classes)) for lvl in range(depth)]

    alpha_cts = [
        ctx.engine.encrypt(alpha0[i].tolist() + [0.0] * (n_pow2_f - n_features), ctx.pk) for i in range(n_internal)
    ]
    threshold_cts = [
        [ctx.engine.encrypt([float(threshold0[i, j])] * slot_count, ctx.pk) for j in range(n_features)]
        for i in range(n_internal)
    ]
    local_logits_cts = [
        [
            ctx.engine.encrypt(local_logits0[lvl][k].tolist() + [0.0] * (n_pow2_c - n_classes), ctx.pk)
            for k in range(1 << (lvl + 1))
        ]
        for lvl in range(depth)
    ]
    return {"alpha": alpha_cts, "threshold": threshold_cts, "local_logits": local_logits_cts}


def decrypt_params_N(ctx, params: dict, n_features: int, n_classes: int, depth: int) -> dict:
    n_internal = (1 << depth) - 1
    alpha = np.array([np.real(ctx.engine.decrypt(params["alpha"][i], ctx.sk))[:n_features] for i in range(n_internal)])
    threshold = np.array(
        [[np.real(ctx.engine.decrypt(params["threshold"][i][j], ctx.sk))[0] for j in range(n_features)] for i in range(n_internal)]
    )
    local_logits = [
        np.array(
            [np.real(ctx.engine.decrypt(params["local_logits"][lvl][k], ctx.sk))[:n_classes] for k in range(1 << (lvl + 1))]
        )
        for lvl in range(depth)
    ]
    return {
        "alpha": alpha, "threshold": threshold, "leaf_logits": local_logits[-1],
        "steepness": STEEPNESS, "depth": depth,
    }


def forward_backward_update_N(ctx, dataset, params: dict, sample_mask, n_features: int, n_classes: int, depth: int, lr: float):
    n_pow2_f = next_power_of_two(n_features)
    n_pow2_c = next_power_of_two(n_classes)
    n_samples = dataset.n_samples

    new_alpha = [None] * ((1 << depth) - 1)
    new_threshold = [None] * ((1 << depth) - 1)
    new_local_logits = [None] * depth

    current_level_probs = [None]
    node_idx = 0
    for level in range(depth):
        start = (1 << level) - 1
        count = 1 << level
        next_level_probs = []
        node_gate = [None] * count
        node_gate_j = [None] * count
        node_w = [None] * count
        node_parent_prob = [None] * count

        # ---- forward: baseline depthN_ckks.py와 100% 동일한 gate 계산(models.gradient_soft_tree.gate 공용 모듈) ----
        for idx, parent_prob in enumerate(current_level_probs):
            i = node_idx
            gate, gate_terms, w = compute_axis_aligned_gate(
                ctx, dataset, params["alpha"][i], params["threshold"][i], n_features, n_pow2_f, min_level=_LOCAL_MIN_LEVEL
            )

            node_gate[idx], node_gate_j[idx], node_w[idx], node_parent_prob[idx] = gate, gate_terms, w, parent_prob

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
        current_level_probs = next_level_probs  # 이 레벨까지의 "가상 leaf" reach prob (길이 2^(level+1))

        # ---- 이 레벨의 local classifier로 바로 loss 계산 (baseline의 leaf loss 블록과
        # 완전히 같은 패턴, params["local_logits"][level]에 대해서만) ----
        n_virtual_leaves = 1 << (level + 1)
        local_dist = [
            packed_softmax(ctx, params["local_logits"][level][k], n_classes, n_pow2_c) for k in range(n_virtual_leaves)
        ]

        y_hat_level = []
        for c in range(n_classes):
            acc = None
            for k in range(n_virtual_leaves):
                ld_c = _ensure_level(ctx, extract_weight_broadcast(ctx, local_dist[k], c))
                term = ctx.engine.multiply(current_level_probs[k], ld_c, ctx.rlk)
                acc = term if acc is None else ctx.engine.add(acc, term)
            y_hat_level.append(_ensure_level(ctx, acc))

        dL_dyhat_level = []
        for c in range(n_classes):
            diff = ctx.engine.subtract(y_hat_level[c], dataset.enc_labels[c])
            dL_dyhat_level.append(_ensure_level(ctx, ctx.engine.multiply(diff, 2.0 / n_samples)))

        g_this_level = [None] * n_virtual_leaves
        local_logits_grad = [None] * n_virtual_leaves
        for k in range(n_virtual_leaves):
            dL_ddist_packed = None
            g_k = None
            for c in range(n_classes):
                term = ctx.engine.multiply(dL_dyhat_level[c], current_level_probs[k], ctx.rlk)
                term = ctx.engine.multiply(term, sample_mask, ctx.rlk)
                term = ctx.engine.intt(term)
                s = _ensure_level(ctx, ctx.engine.sum(term, ctx.rotation_key))
                piece = scatter_to_slot(ctx, s, c)
                dL_ddist_packed = piece if dL_ddist_packed is None else ctx.engine.add(dL_ddist_packed, piece)

                ld_c = extract_weight_broadcast(ctx, local_dist[k], c)
                g_piece = ctx.engine.multiply(dL_dyhat_level[c], ld_c, ctx.rlk)
                g_k = g_piece if g_k is None else ctx.engine.add(g_k, g_piece)
            local_logits_grad[k] = softmax_backward_packed(ctx, local_dist[k], _ensure_level(ctx, dL_ddist_packed), n_classes, n_pow2_c)
            g_this_level[k] = _ensure_level(ctx, g_k)
            gc.collect()

        new_local_logits[level] = [
            _ensure_level(
                ctx, ctx.engine.subtract(params["local_logits"][level][k], ctx.engine.multiply(local_logits_grad[k], lr))
            )
            for k in range(n_virtual_leaves)
        ]

        # ---- 이 레벨의 gate 파라미터는 g_this_level에서만 gradient를 받는다(baseline의
        # backward 레벨 루프 한 번과 완전히 같은 공식, 단 next_g로 다음 레벨에 넘기지 않음) ----
        for idx in range(count):
            i = start + idx
            g_left, g_right = g_this_level[2 * idx], g_this_level[2 * idx + 1]
            gate, w, gate_terms, p_i = node_gate[idx], node_w[idx], node_gate_j[idx], node_parent_prob[idx]

            diff_g = _ensure_level(ctx, ctx.engine.subtract(g_right, g_left))
            if p_i is None:
                dL_dgate_i = diff_g
            else:
                dL_dgate_i = _ensure_level(ctx, ctx.engine.multiply(p_i, diff_g, ctx.rlk))

            dL_dalpha_packed = None
            new_thresh_i = []
            for j in range(n_features):
                gate_j = _ensure_level(ctx, gate_terms[j])
                surrogate = _ensure_level(ctx, ctx.engine.multiply(gate_j, ctx.engine.subtract(1.0, gate_j), ctx.rlk))
                w_j = _ensure_level(ctx, extract_weight_broadcast(ctx, w, j))

                prod_t = _ensure_level(ctx, ctx.engine.multiply(dL_dgate_i, surrogate, ctx.rlk))
                prod_t = ctx.engine.multiply(prod_t, sample_mask, ctx.rlk)
                prod_t = ctx.engine.intt(prod_t)
                sum_t = _ensure_level(ctx, ctx.engine.sum(prod_t, ctx.rotation_key))
                dL_dt_j = _ensure_level(ctx, ctx.engine.multiply(sum_t, w_j, ctx.rlk))
                dL_dt_j = ctx.engine.multiply(dL_dt_j, -STEEPNESS)
                new_thresh_i.append(
                    _ensure_level(ctx, ctx.engine.subtract(params["threshold"][i][j], ctx.engine.multiply(dL_dt_j, lr)))
                )

                term_a = _ensure_level(ctx, ctx.engine.multiply(dL_dgate_i, ctx.engine.subtract(gate_j, gate), ctx.rlk))
                term_a = ctx.engine.multiply(term_a, sample_mask, ctx.rlk)
                term_a = ctx.engine.intt(term_a)
                sum_a = _ensure_level(ctx, ctx.engine.sum(term_a, ctx.rotation_key))
                piece_a = scatter_to_slot(ctx, sum_a, j)
                dL_dalpha_packed = piece_a if dL_dalpha_packed is None else ctx.engine.add(dL_dalpha_packed, piece_a)
                gc.collect()

            dL_dalpha_packed = _ensure_level(ctx, dL_dalpha_packed)
            w_lvl = _ensure_level(ctx, w)
            dL_dalpha_i = ctx.engine.multiply(w_lvl, dL_dalpha_packed, ctx.rlk)
            new_alpha[i] = _ensure_level(ctx, ctx.engine.subtract(params["alpha"][i], ctx.engine.multiply(dL_dalpha_i, lr)))
            new_threshold[i] = new_thresh_i
        gc.collect()

    return {"alpha": new_alpha, "threshold": new_threshold, "local_logits": new_local_logits}


def _debug_validate(dataset_name: str, depth: int, n_epochs: int, lr: float, seed: int = 0, level_preset: int | None = None):
    """baseline depthN_ckks.py의 _debug_validate와 같은 목적 - 단일 프로세스로 몇 epoch만
    돌려 plaintext(local_loss/depthN_reference)와 대조."""
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
        ref = train_depthN_local_loss(X_train, y_train_oh, depth=depth, lr=lr, epochs=epoch, seed=seed)
        max_err = max(
            np.abs(decoded["alpha"] - ref["alpha"]).max(),
            np.abs(decoded["threshold"] - ref["threshold"]).max(),
            np.abs(decoded["leaf_logits"] - ref["leaf_logits"]).max(),
        )
        print(f"[epoch {epoch}] elapsed={elapsed:.1f}s  max abs diff vs plaintext = {max_err:.5f}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "iris"
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    lr_ = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    lp = int(sys.argv[5]) if len(sys.argv) > 5 else None
    _debug_validate(ds, d, n_ep, lr_, level_preset=lp)
