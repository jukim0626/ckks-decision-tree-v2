"""depth1_ckks.py의 forward/backward를 임의 depth로 일반화 - depthN_reference.py에서
유도한 재귀 관계(leaf/node의 reach probability에 대한 g=dL/dp를 leaf에서 root 방향으로
전파)를 그대로 암호문 연산으로 옮긴 것.

depth=1에서 이미 CKKS로 검증된(plaintext 대비 오차 0.0006) threshold/alpha gradient
공식을 노드마다 재사용한다 - 새로 근사하거나 새로 설계한 부분은 없고, depth1_ckks.py가
"레벨이 1개뿐인 특수 케이스"였던 걸 레벨 루프로 풀었을 뿐이다."""

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
from models.gradient_soft_tree.baseline.depthN_reference import train_depthN  # noqa: E402

# depth1_ckks.py의 _LOCAL_MIN_LEVEL과 동일한 이유/값 - closed_form_mgi의 min_level=8 기본값은
# 안 건드리고 이 실험 파일 안에서만 낮은 임계값을 쓴다.
_LOCAL_MIN_LEVEL = 5


def _ensure_level(ctx, ct):
    return ensure_level(ctx, ct, min_level=_LOCAL_MIN_LEVEL)


def init_encrypted_params_N(ctx, n_features: int, n_classes: int, depth: int, seed: int, slot_count: int):
    """depthN_reference.train_depthN과 완전히 같은 초기화(같은 seed) - 두 트랙 비교용."""
    rng = np.random.default_rng(seed)
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth
    n_pow2_f = next_power_of_two(n_features)
    n_pow2_c = next_power_of_two(n_classes)

    alpha0 = rng.normal(0, 0.1, size=(n_internal, n_features))
    threshold0 = rng.normal(0, 0.1, size=(n_internal, n_features))
    leaf0 = rng.normal(0, 0.1, size=(n_leaves, n_classes))

    alpha_cts = [
        ctx.engine.encrypt(alpha0[i].tolist() + [0.0] * (n_pow2_f - n_features), ctx.pk) for i in range(n_internal)
    ]
    threshold_cts = [
        [ctx.engine.encrypt([float(threshold0[i, j])] * slot_count, ctx.pk) for j in range(n_features)]
        for i in range(n_internal)
    ]
    leaf_cts = [
        ctx.engine.encrypt(leaf0[l].tolist() + [0.0] * (n_pow2_c - n_classes), ctx.pk) for l in range(n_leaves)
    ]
    return {"alpha": alpha_cts, "threshold": threshold_cts, "leaf_logits": leaf_cts}


def decrypt_params_N(ctx, params: dict, n_features: int, n_classes: int, depth: int) -> dict:
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth
    alpha = np.array([np.real(ctx.engine.decrypt(params["alpha"][i], ctx.sk))[:n_features] for i in range(n_internal)])
    threshold = np.array(
        [[np.real(ctx.engine.decrypt(params["threshold"][i][j], ctx.sk))[0] for j in range(n_features)] for i in range(n_internal)]
    )
    leaf_logits = np.array(
        [np.real(ctx.engine.decrypt(params["leaf_logits"][l], ctx.sk))[:n_classes] for l in range(n_leaves)]
    )
    return {"alpha": alpha, "threshold": threshold, "leaf_logits": leaf_logits, "steepness": STEEPNESS, "depth": depth}


def forward_backward_update_N(ctx, dataset, params: dict, sample_mask, n_features: int, n_classes: int, depth: int, lr: float):
    n_pow2_f = next_power_of_two(n_features)
    n_pow2_c = next_power_of_two(n_classes)
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth

    node_gate = [None] * n_internal
    node_gate_j = [None] * n_internal
    node_w = [None] * n_internal
    node_parent_prob = [None] * n_internal  # None == root(암묵적으로 1)

    current_level_probs = [None]
    node_idx = 0
    for _level in range(depth):
        next_level_probs = []
        for parent_prob in current_level_probs:
            i = node_idx
            gate, gate_terms, w = compute_axis_aligned_gate(
                ctx, dataset, params["alpha"][i], params["threshold"][i], n_features, n_pow2_f, min_level=_LOCAL_MIN_LEVEL
            )

            node_gate[i], node_gate_j[i], node_w[i], node_parent_prob[i] = gate, gate_terms, w, parent_prob

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

    leafdist = [packed_softmax(ctx, params["leaf_logits"][l], n_classes, n_pow2_c) for l in range(n_leaves)]

    y_hat = []
    for c in range(n_classes):
        acc = None
        for l in range(n_leaves):
            ld_c = _ensure_level(ctx, extract_weight_broadcast(ctx, leafdist[l], c))
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

            ld_c = extract_weight_broadcast(ctx, leafdist[l], c)
            g_piece = ctx.engine.multiply(dL_dyhat[c], ld_c, ctx.rlk)
            g_l = g_piece if g_l is None else ctx.engine.add(g_l, g_piece)
        dL_dleaf_logits[l] = softmax_backward_packed(ctx, leafdist[l], _ensure_level(ctx, dL_ddist_packed), n_classes, n_pow2_c)
        current_g[l] = _ensure_level(ctx, g_l)
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
            gate, w, gate_terms, p_i = node_gate[i], node_w[i], node_gate_j[i], node_parent_prob[i]

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
        current_g = next_g
        gc.collect()

    new_leaf_logits = [
        _ensure_level(ctx, ctx.engine.subtract(params["leaf_logits"][l], ctx.engine.multiply(dL_dleaf_logits[l], lr)))
        for l in range(n_leaves)
    ]

    return {"alpha": new_alpha, "threshold": new_threshold, "leaf_logits": new_leaf_logits}


def _debug_validate(dataset_name: str, depth: int, n_epochs: int, lr: float, seed: int = 0, level_preset: int | None = None):
    """단일 프로세스로 몇 epoch만 돌려 plaintext(depthN_reference)와 대조하는 디버그
    진입점 - 여러 epoch을 실제로 돌리려면 train_depthN_ckks.py(epoch별 프로세스 격리)를 쓸 것."""
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
        import time

        t0 = time.time()
        params = forward_backward_update_N(ctx, dataset, params, sample_mask, n_features, n_classes, depth, lr=lr)
        elapsed = time.time() - t0
        decoded = decrypt_params_N(ctx, params, n_features, n_classes, depth)
        ref = train_depthN(X_train, y_train_oh, depth=depth, lr=lr, epochs=epoch, seed=seed)
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
