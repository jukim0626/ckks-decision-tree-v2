"""baseline forward_backward_update_N와 packed forward_backward_update_N_packed을 각각
1 epoch 실제로 돌리되, 내부에서 dL_dgate_i(threshold/attention gradient의 공통 입력)를
decrypt해서 직접 비교 - debug_compare_sum_t.py는 더미 dL_dgate_i로는 threshold 경로 자체가
정확함을 확인했으므로, 이번엔 "진짜" dL_dgate_i가 baseline과 packed 사이에서 이미
어긋나는지를 확인한다. depthN_ckks.forward_backward_update_N을 그대로 복붙하되 마지막에
dL_dgate_i를 decrypt해서 리턴하도록 살짝만 바꾼 로컬 버전을 쓴다(원본 파일은 안 건드림)."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.approximation.sigmoid import sigmoid_approx_enc  # noqa: E402
from core.data.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from core.ckks_engine import create_bootstrap_context, ensure_level  # noqa: E402
from core.encrypted_ops.slot_packing import next_power_of_two, scatter_to_slot  # noqa: E402
from core.encrypted_ops.slot_packing import extract_weight_broadcast as _extract_weight_broadcast  # noqa: E402
from models.gradient_soft_tree.depth1_ckks import STEEPNESS, packed_softmax, softmax_backward_packed  # noqa: E402
from models.gradient_soft_tree.depthN_ckks import init_encrypted_params_N  # noqa: E402
from models.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    assert_layout_fits,
    broadcast_full_to_blocks,
    build_block_masks,
    compute_block_size,
    extract_block_to_full,
    pack_dataset_features_blocked,
    pack_threshold_blocked,
)

_LOCAL_MIN_LEVEL = 5


def _lvl(ctx, ct):
    return ensure_level(ctx, ct, min_level=_LOCAL_MIN_LEVEL)


def dgate_baseline(ctx, dataset, params, sample_mask, n_features, n_classes, depth):
    n_pow2_f = next_power_of_two(n_features)
    n_pow2_c = next_power_of_two(n_classes)
    n_leaves = 1 << depth
    i = 0
    w = packed_softmax(ctx, params["alpha"][i], n_features, n_pow2_f)
    gate = None
    for j in range(n_features):
        enc_feature = _lvl(ctx, dataset.enc_features[j])
        threshold_j = _lvl(ctx, params["threshold"][i][j])
        enc_diff = ctx.engine.subtract(enc_feature, threshold_j)
        enc_diff = ensure_level(ctx, enc_diff, min_level=12)
        gate_j = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff)
        w_j = _extract_weight_broadcast(ctx, w, j)
        piece = ctx.engine.multiply(w_j, gate_j, ctx.rlk)
        gate = piece if gate is None else ctx.engine.add(gate, piece)
    gate = _lvl(ctx, gate)
    left = ctx.engine.subtract(1.0, gate)
    right = gate
    leaf_probs = [_lvl(ctx, left), _lvl(ctx, right)]

    leafdist = [packed_softmax(ctx, params["leaf_logits"][l], n_classes, n_pow2_c) for l in range(n_leaves)]
    y_hat = []
    for c in range(n_classes):
        acc = None
        for l in range(n_leaves):
            ld_c = _lvl(ctx, _extract_weight_broadcast(ctx, leafdist[l], c))
            term = ctx.engine.multiply(leaf_probs[l], ld_c, ctx.rlk)
            acc = term if acc is None else ctx.engine.add(acc, term)
        y_hat.append(_lvl(ctx, acc))
    n_samples = dataset.n_samples
    dL_dyhat = [
        _lvl(ctx, ctx.engine.multiply(ctx.engine.subtract(y_hat[c], dataset.enc_labels[c]), 2.0 / n_samples))
        for c in range(n_classes)
    ]
    current_g = [None] * n_leaves
    for l in range(n_leaves):
        g_l = None
        for c in range(n_classes):
            ld_c = _extract_weight_broadcast(ctx, leafdist[l], c)
            g_piece = ctx.engine.multiply(dL_dyhat[c], ld_c, ctx.rlk)
            g_l = g_piece if g_l is None else ctx.engine.add(g_l, g_piece)
        current_g[l] = _lvl(ctx, g_l)
    diff_g = _lvl(ctx, ctx.engine.subtract(current_g[1], current_g[0]))
    return diff_g  # root, p_i=None -> dL_dgate_i == diff_g


def dgate_packed(ctx, dataset, params, sample_mask, blocked_features, sample_mask_blocked, block_masks, block_size, n_features, n_classes, depth):
    n_pow2_f = next_power_of_two(n_features)
    n_pow2_c = next_power_of_two(n_classes)
    n_leaves = 1 << depth
    i = 0
    w = packed_softmax(ctx, params["alpha"][i], n_features, n_pow2_f)

    blocked_threshold_i = pack_threshold_blocked(ctx, params["threshold"][i], block_masks)
    enc_diff_blocked = ctx.engine.subtract(blocked_features, blocked_threshold_i)
    enc_diff_blocked = ensure_level(ctx, enc_diff_blocked, min_level=12)
    gate_blocked = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff_blocked)

    gate = None
    for j in range(n_features):
        gate_j = extract_block_to_full(ctx, gate_blocked, j, block_size, sample_mask)
        w_j = _extract_weight_broadcast(ctx, w, j)
        piece = ctx.engine.multiply(w_j, gate_j, ctx.rlk)
        gate = piece if gate is None else ctx.engine.add(gate, piece)
    gate = _lvl(ctx, gate)
    left = ctx.engine.subtract(1.0, gate)
    right = gate
    leaf_probs = [_lvl(ctx, left), _lvl(ctx, right)]

    leafdist = [packed_softmax(ctx, params["leaf_logits"][l], n_classes, n_pow2_c) for l in range(n_leaves)]
    y_hat = []
    for c in range(n_classes):
        acc = None
        for l in range(n_leaves):
            ld_c = _lvl(ctx, _extract_weight_broadcast(ctx, leafdist[l], c))
            term = ctx.engine.multiply(leaf_probs[l], ld_c, ctx.rlk)
            acc = term if acc is None else ctx.engine.add(acc, term)
        y_hat.append(_lvl(ctx, acc))
    n_samples = dataset.n_samples
    dL_dyhat = [
        _lvl(ctx, ctx.engine.multiply(ctx.engine.subtract(y_hat[c], dataset.enc_labels[c]), 2.0 / n_samples))
        for c in range(n_classes)
    ]
    current_g = [None] * n_leaves
    for l in range(n_leaves):
        g_l = None
        for c in range(n_classes):
            ld_c = _extract_weight_broadcast(ctx, leafdist[l], c)
            g_piece = ctx.engine.multiply(dL_dyhat[c], ld_c, ctx.rlk)
            g_l = g_piece if g_l is None else ctx.engine.add(g_l, g_piece)
        current_g[l] = _lvl(ctx, g_l)
    diff_g = _lvl(ctx, ctx.engine.subtract(current_g[1], current_g[0]))
    return diff_g


def main():
    dataset_name, depth, seed, level_preset = "iris", 1, 0, 17
    X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name)
    n_features = X_train.shape[1]
    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    y_train_oh = one_hot_encode(y_train, n_classes)

    ctx = create_bootstrap_context(mode="gpu", level_preset=level_preset)
    dataset = encrypt_dataset(ctx, X_train, y_train_oh)
    sample_mask = ctx.engine.encrypt([1.0] * dataset.n_samples, ctx.pk)

    block_size = compute_block_size(dataset.n_samples)
    assert_layout_fits(n_features, block_size, ctx.engine.slot_count)
    block_masks = build_block_masks(n_features, block_size, ctx.engine.slot_count)
    blocked_features = pack_dataset_features_blocked(ctx, dataset.enc_features, block_size)
    sample_mask_blocked = broadcast_full_to_blocks(ctx, sample_mask, n_features, block_size)

    params_b = init_encrypted_params_N(ctx, n_features, n_classes, depth, seed=seed, slot_count=ctx.engine.slot_count)
    params_p = init_encrypted_params_N(ctx, n_features, n_classes, depth, seed=seed, slot_count=ctx.engine.slot_count)

    dg_b = dgate_baseline(ctx, dataset, params_b, sample_mask, n_features, n_classes, depth)
    dg_p = dgate_packed(ctx, dataset, params_p, sample_mask, blocked_features, sample_mask_blocked, block_masks, block_size, n_features, n_classes, depth)

    n = dataset.n_samples
    vb = np.real(ctx.engine.decrypt(dg_b, ctx.sk))[:n]
    vp = np.real(ctx.engine.decrypt(dg_p, ctx.sk))[:n]
    print("dL_dgate_i baseline[:10]:", np.round(vb[:10], 5))
    print("dL_dgate_i packed  [:10]:", np.round(vp[:10], 5))
    print("max abs diff (real region):", np.abs(vb - vp).max())

    vb_full = np.real(ctx.engine.decrypt(dg_b, ctx.sk))
    vp_full = np.real(ctx.engine.decrypt(dg_p, ctx.sk))
    print("baseline far-field (slot 5000, 20000):", vb_full[5000], vb_full[20000])
    print("packed far-field (slot 5000, 20000):", vp_full[5000], vp_full[20000])


if __name__ == "__main__":
    main()
