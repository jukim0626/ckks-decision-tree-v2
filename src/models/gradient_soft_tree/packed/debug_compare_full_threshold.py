"""baseline과 packed 각각의 "진짜" forward pass에서 나온 진짜 dL_dgate_i/gate_terms(또는
gate_blocked)/w를 그대로 이어서 threshold gradient(dL_dt_j)까지 계산해 feature별로 직접
비교한다. debug_compare_dgate.py(dL_dgate_i만 비교, 거의 일치 확인)와
debug_compare_sum_t.py(더미 dL_dgate_i로 threshold 파이프라인만 비교, 정확히 일치 확인)를
합쳐서 "진짜 값 두 개를 진짜 파이프라인에 넣었을 때"만 재현되는 차이를 잡아낸다."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.approximation.sigmoid import sigmoid_approx_enc  # noqa: E402
from core.data.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from core.ckks_engine import create_bootstrap_context, ensure_level  # noqa: E402
from core.encrypted_ops.slot_packing import next_power_of_two  # noqa: E402
from core.encrypted_ops.slot_packing import extract_weight_broadcast as _extract_weight_broadcast  # noqa: E402
from models.gradient_soft_tree.depth1_ckks import STEEPNESS, packed_softmax  # noqa: E402
from models.gradient_soft_tree.depthN_ckks import init_encrypted_params_N  # noqa: E402
from models.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    assert_layout_fits,
    block_local_sum,
    broadcast_full_to_blocks,
    build_block_masks,
    compute_block_size,
    extract_block_to_full,
    gather_block_tops_to_packed,
    pack_dataset_features_blocked,
    pack_threshold_blocked,
)

_LOCAL_MIN_LEVEL = 5


def _lvl(ctx, ct):
    return ensure_level(ctx, ct, min_level=_LOCAL_MIN_LEVEL)


def run_baseline(ctx, dataset, params, sample_mask, n_features, n_classes, depth):
    n_pow2_f = next_power_of_two(n_features)
    n_pow2_c = next_power_of_two(n_classes)
    n_leaves = 1 << depth
    i = 0
    w = packed_softmax(ctx, params["alpha"][i], n_features, n_pow2_f)
    gate_terms = []
    gate = None
    for j in range(n_features):
        enc_feature = _lvl(ctx, dataset.enc_features[j])
        threshold_j = _lvl(ctx, params["threshold"][i][j])
        enc_diff = ctx.engine.subtract(enc_feature, threshold_j)
        enc_diff = ensure_level(ctx, enc_diff, min_level=12)
        gate_j = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff)
        gate_terms.append(gate_j)
        w_j = _extract_weight_broadcast(ctx, w, j)
        piece = ctx.engine.multiply(w_j, gate_j, ctx.rlk)
        gate = piece if gate is None else ctx.engine.add(gate, piece)
    gate = _lvl(ctx, gate)
    leaf_probs = [_lvl(ctx, ctx.engine.subtract(1.0, gate)), _lvl(ctx, gate)]

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
    dL_dgate_i = _lvl(ctx, ctx.engine.subtract(current_g[1], current_g[0]))
    print(f"[baseline] dL_dgate_i level={dL_dgate_i.level}  decrypted[:5]={np.round(np.real(ctx.engine.decrypt(dL_dgate_i, ctx.sk))[:5], 6)}")

    dL_dt = []
    for j in range(n_features):
        gate_j = _lvl(ctx, gate_terms[j])
        print(f"[baseline] feature {j} gate_j level={gate_j.level}")
        surrogate = _lvl(ctx, ctx.engine.multiply(gate_j, ctx.engine.subtract(1.0, gate_j), ctx.rlk))
        prod_t = ctx.engine.multiply(dL_dgate_i, surrogate, ctx.rlk)
        prod_t = ctx.engine.multiply(prod_t, sample_mask, ctx.rlk)
        prod_t = ctx.engine.intt(prod_t)
        sum_t = _lvl(ctx, ctx.engine.sum(prod_t, ctx.rotation_key))
        print(f"[baseline] feature {j} sum_t={np.real(ctx.engine.decrypt(sum_t, ctx.sk))[0]:.6f}")
        w_j = _lvl(ctx, _extract_weight_broadcast(ctx, w, j))
        dL_dt_j = _lvl(ctx, ctx.engine.multiply(sum_t, w_j, ctx.rlk))
        dL_dt_j = ctx.engine.multiply(dL_dt_j, -STEEPNESS)
        dL_dt.append(np.real(ctx.engine.decrypt(dL_dt_j, ctx.sk))[0])
    return dL_dt


def run_packed(ctx, dataset, params, sample_mask, blocked_features, sample_mask_blocked, block_masks, block_size, n_features, n_classes, depth):
    n_pow2_f = next_power_of_two(n_features)
    n_pow2_c = next_power_of_two(n_classes)
    n_leaves = 1 << depth
    slot_count = ctx.engine.slot_count
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
    leaf_probs = [_lvl(ctx, ctx.engine.subtract(1.0, gate)), _lvl(ctx, gate)]

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
    dL_dgate_i = _lvl(ctx, ctx.engine.subtract(current_g[1], current_g[0]))
    print(f"[packed]   dL_dgate_i level={dL_dgate_i.level}  decrypted[:5]={np.round(np.real(ctx.engine.decrypt(dL_dgate_i, ctx.sk))[:5], 6)}")
    print(f"[packed]   dL_dgate_i far-field (slot 5000): {np.real(ctx.engine.decrypt(dL_dgate_i, ctx.sk))[5000]:.6f}")
    dL_dgate_i_masked = ctx.engine.multiply(dL_dgate_i, sample_mask, ctx.rlk)  # 픽스: broadcast 전에 마스킹

    gate_blocked_lvl = _lvl(ctx, gate_blocked)
    surrogate_blocked = _lvl(ctx, ctx.engine.multiply(gate_blocked_lvl, ctx.engine.subtract(1.0, gate_blocked_lvl), ctx.rlk))
    dL_dgate_i_blocked = broadcast_full_to_blocks(ctx, dL_dgate_i_masked, n_features, block_size)
    dec_orig = np.real(ctx.engine.decrypt(dL_dgate_i, ctx.sk))
    dec_blocked = np.real(ctx.engine.decrypt(dL_dgate_i_blocked, ctx.sk))
    n = dataset.n_samples
    for j in range(n_features):
        region = dec_blocked[j * block_size : j * block_size + n]
        d = np.abs(region - dec_orig[:n]).max()
        print(f"[packed]   broadcast_full_to_blocks check feature {j}: max_abs_diff_vs_orig={d:.6f}  region[:3]={np.round(region[:3],6)}  orig[:3]={np.round(dec_orig[:3],6)}")
    prod_t_blocked = ctx.engine.multiply(dL_dgate_i_blocked, surrogate_blocked, ctx.rlk)
    prod_t_blocked = ctx.engine.multiply(prod_t_blocked, sample_mask_blocked, ctx.rlk)
    prod_t_blocked = ctx.engine.intt(prod_t_blocked)
    sum_t_blocked = block_local_sum(ctx, prod_t_blocked, block_size)
    sum_t_blocked = _lvl(ctx, sum_t_blocked)
    sum_t_packed = gather_block_tops_to_packed(ctx, sum_t_blocked, n_features, block_size, slot_count)
    print(f"[packed]   sum_t_packed[:4]={np.round(np.real(ctx.engine.decrypt(sum_t_packed, ctx.sk))[:4], 6)}")
    w_for_t = _lvl(ctx, w)
    dL_dt_packed = ctx.engine.multiply(sum_t_packed, w_for_t, ctx.rlk)
    dL_dt_packed = ctx.engine.multiply(dL_dt_packed, -STEEPNESS)
    dL_dt_packed = _lvl(ctx, dL_dt_packed)

    dL_dt = []
    for j in range(n_features):
        dL_dt_j = _extract_weight_broadcast(ctx, dL_dt_packed, j)
        dL_dt.append(np.real(ctx.engine.decrypt(dL_dt_j, ctx.sk))[0])
    return dL_dt


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

    dL_dt_baseline = run_baseline(ctx, dataset, params_b, sample_mask, n_features, n_classes, depth)
    dL_dt_packed = run_packed(ctx, dataset, params_p, sample_mask, blocked_features, sample_mask_blocked, block_masks, block_size, n_features, n_classes, depth)

    for j in range(n_features):
        print(f"feature {j}: baseline={dL_dt_baseline[j]:.6f}  packed={dL_dt_packed[j]:.6f}  diff={dL_dt_packed[j]-dL_dt_baseline[j]:.6f}")


if __name__ == "__main__":
    main()
