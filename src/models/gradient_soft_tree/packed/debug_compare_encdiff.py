"""가장 기초 단계: enc_diff(=feature-threshold, sigmoid 들어가기 직전)를 baseline
per-feature 방식과 packed block 방식에서 각각 decrypt해서 feature별로 직접 비교. 실제
iris 데이터/실제 초기 threshold로, sigmoid조차 태우기 전 단계에서 이미 어긋나는지 확인."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.approximation.sigmoid import sigmoid_approx_enc  # noqa: E402
from core.data.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from core.ckks_engine import create_bootstrap_context, ensure_level  # noqa: E402
from models.gradient_soft_tree.baseline.depthN_ckks import init_encrypted_params_N  # noqa: E402
from models.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    assert_layout_fits,
    build_block_masks,
    compute_block_size,
    extract_block_to_full,
    pack_dataset_features_blocked,
    pack_threshold_blocked,
)

_LOCAL_MIN_LEVEL = 5


def _lvl(ctx, ct):
    return ensure_level(ctx, ct, min_level=_LOCAL_MIN_LEVEL)


def main():
    dataset_name, depth, seed, level_preset = "iris", 1, 0, 17
    X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name)
    n_features = X_train.shape[1]
    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    y_train_oh = one_hot_encode(y_train, n_classes)

    ctx = create_bootstrap_context(mode="gpu", level_preset=level_preset)
    dataset = encrypt_dataset(ctx, X_train, y_train_oh)

    block_size = compute_block_size(dataset.n_samples)
    assert_layout_fits(n_features, block_size, ctx.engine.slot_count)
    block_masks = build_block_masks(n_features, block_size, ctx.engine.slot_count)
    blocked_features = pack_dataset_features_blocked(ctx, dataset.enc_features, block_size)

    params = init_encrypted_params_N(ctx, n_features, n_classes, depth, seed=seed, slot_count=ctx.engine.slot_count)
    i = 0

    blocked_threshold_i = pack_threshold_blocked(ctx, params["threshold"][i], block_masks)
    enc_diff_blocked = ctx.engine.subtract(blocked_features, blocked_threshold_i)
    dec_blocked = np.real(ctx.engine.decrypt(enc_diff_blocked, ctx.sk))
    print(f"enc_diff_blocked level BEFORE ensure_level(12): {enc_diff_blocked.level}")
    enc_diff_blocked_lvl = ensure_level(ctx, enc_diff_blocked, min_level=12)
    print(f"enc_diff_blocked level AFTER ensure_level(12): {enc_diff_blocked_lvl.level}")
    gate_blocked = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff_blocked_lvl)
    sample_mask = ctx.engine.encrypt([1.0] * dataset.n_samples, ctx.pk)

    n = dataset.n_samples
    for j in range(n_features):
        enc_feature = _lvl(ctx, dataset.enc_features[j])
        threshold_j = _lvl(ctx, params["threshold"][i][j])
        enc_diff = ctx.engine.subtract(enc_feature, threshold_j)
        dec_baseline = np.real(ctx.engine.decrypt(enc_diff, ctx.sk))[:n]

        block_region = dec_blocked[j * block_size : j * block_size + n]
        diff = np.abs(dec_baseline - block_region)
        print(f"feature {j} enc_diff: max_abs_diff={diff.max():.6f}")

        print(f"  feature {j} threshold level: {params['threshold'][i][j].level}")
        enc_diff_j = ensure_level(ctx, enc_diff, min_level=12)
        print(f"  feature {j} enc_diff level BEFORE ensure_level(12): {enc_diff.level}, AFTER: {enc_diff_j.level}")
        gate_j_baseline = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff_j)
        dec_gate_baseline = np.real(ctx.engine.decrypt(gate_j_baseline, ctx.sk))[:n]

        gate_j_packed = extract_block_to_full(ctx, gate_blocked, j, block_size, sample_mask)
        dec_gate_packed = np.real(ctx.engine.decrypt(gate_j_packed, ctx.sk))[:n]

        gdiff = np.abs(dec_gate_baseline - dec_gate_packed)
        print(f"  feature {j} gate: max_abs_diff={gdiff.max():.6f}  baseline[:5]={np.round(dec_gate_baseline[:5],5)}  packed[:5]={np.round(dec_gate_packed[:5],5)}")


if __name__ == "__main__":
    main()
