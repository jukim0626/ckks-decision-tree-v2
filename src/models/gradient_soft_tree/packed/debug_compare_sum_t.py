"""threshold gradient 경로에서 baseline의 per-feature sum_t와 packed의
gather_block_tops_to_packed(block_local_sum(...))에서 뽑아낸 값을 feature별로 직접
decrypt해서 비교 - 1-epoch 전체 비교(debug_compare_single_epoch.py)에서 threshold만
유독 오차가 큰 이유를 좁히기 위한 디버그 전용 스크립트. 두 트랙이 같은 초기 params/
dataset에서 시작해서, forward는 baseline 그대로 한 번 태우고 그 결과(gate/w/dL_dgate_i 등)를
공유한 뒤 threshold 관련 중간값만 두 가지 방식으로 각각 계산해서 비교한다."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from core.ckks_engine import create_bootstrap_context, ensure_level  # noqa: E402
from core.encrypted_ops.slot_packing import extract_weight_broadcast as _extract_weight_broadcast  # noqa: E402
from models.gradient_soft_tree.baseline.depth1_ckks import STEEPNESS, packed_softmax  # noqa: E402
from models.gradient_soft_tree.baseline.depthN_ckks import init_encrypted_params_N  # noqa: E402
from core.approximation.sigmoid import sigmoid_approx_enc  # noqa: E402
from core.encrypted_ops.slot_packing import next_power_of_two  # noqa: E402
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


def main():
    dataset_name, depth, lr, seed, level_preset = "iris", 1, 2.0, 0, 17
    X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name)
    n_features = X_train.shape[1]
    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    y_train_oh = one_hot_encode(y_train, n_classes)
    n_pow2_f = next_power_of_two(n_features)

    ctx = create_bootstrap_context(mode="gpu", level_preset=level_preset)
    dataset = encrypt_dataset(ctx, X_train, y_train_oh)
    sample_mask = ctx.engine.encrypt([1.0] * dataset.n_samples, ctx.pk)

    block_size = compute_block_size(dataset.n_samples)
    assert_layout_fits(n_features, block_size, ctx.engine.slot_count)
    block_masks = build_block_masks(n_features, block_size, ctx.engine.slot_count)
    blocked_features = pack_dataset_features_blocked(ctx, dataset.enc_features, block_size)
    sample_mask_blocked = broadcast_full_to_blocks(ctx, sample_mask, n_features, block_size)

    params = init_encrypted_params_N(ctx, n_features, n_classes, depth, seed=seed, slot_count=ctx.engine.slot_count)
    i = 0  # depth=1 root

    # --- forward: baseline 방식(per-feature)과 packed 방식을 둘 다 계산해서 gate_terms/gate_blocked를 얻는다 ---
    w = packed_softmax(ctx, params["alpha"][i], n_features, n_pow2_f)

    gate_terms = []
    gate_baseline = None
    for j in range(n_features):
        enc_feature = _lvl(ctx, dataset.enc_features[j])
        threshold_j = _lvl(ctx, params["threshold"][i][j])
        enc_diff = ctx.engine.subtract(enc_feature, threshold_j)
        enc_diff = ensure_level(ctx, enc_diff, min_level=12)
        gate_j = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff)
        gate_terms.append(gate_j)
        w_j = _extract_weight_broadcast(ctx, w, j)
        piece = ctx.engine.multiply(w_j, gate_j, ctx.rlk)
        gate_baseline = piece if gate_baseline is None else ctx.engine.add(gate_baseline, piece)
    gate_baseline = _lvl(ctx, gate_baseline)

    blocked_threshold_i = pack_threshold_blocked(ctx, params["threshold"][i], block_masks)
    enc_diff_blocked = ctx.engine.subtract(blocked_features, blocked_threshold_i)
    enc_diff_blocked = ensure_level(ctx, enc_diff_blocked, min_level=12)
    gate_blocked = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff_blocked)

    # baseline sum_t를 구하려면 dL_dgate_i가 필요한데, 이건 leaf backward를 거쳐야 나오는 값이라
    # 여기서는 대신 "임의의 dL_dgate_i 역할을 하는 더미 값"으로 sample_mask 자체를 씀
    # (분석 목적: dL_dgate_i가 뭐든 threshold gradient 공식의 구조적 정합성만 확인하면 됨).
    dL_dgate_i = sample_mask  # dummy: 실제 gradient가 아니라 baseline/packed 두 경로 비교용

    print("--- baseline 방식: feature별 sum_t -> dL_dt_j ---")
    sum_t_baseline = []
    dL_dt_baseline = []
    for j in range(n_features):
        gate_j = _lvl(ctx, gate_terms[j])
        surrogate = _lvl(ctx, ctx.engine.multiply(gate_j, ctx.engine.subtract(1.0, gate_j), ctx.rlk))
        prod_t = ctx.engine.multiply(dL_dgate_i, surrogate, ctx.rlk)
        prod_t = ctx.engine.multiply(prod_t, sample_mask, ctx.rlk)
        prod_t = ctx.engine.intt(prod_t)
        sum_t = _lvl(ctx, ctx.engine.sum(prod_t, ctx.rotation_key))
        val = np.real(ctx.engine.decrypt(sum_t, ctx.sk))[0]
        sum_t_baseline.append(val)

        w_j = _lvl(ctx, _extract_weight_broadcast(ctx, w, j))
        dL_dt_j = _lvl(ctx, ctx.engine.multiply(sum_t, w_j, ctx.rlk))
        dL_dt_j = ctx.engine.multiply(dL_dt_j, -STEEPNESS)
        dval = np.real(ctx.engine.decrypt(dL_dt_j, ctx.sk))[0]
        dL_dt_baseline.append(dval)
        print(f"  feature {j}: sum_t={val:.6f}  dL_dt_j={dval:.6f}")

    print("--- packed 방식: block_local_sum + gather -> dL_dt_packed ---")
    gate_blocked_lvl = _lvl(ctx, gate_blocked)
    surrogate_blocked = _lvl(ctx, ctx.engine.multiply(gate_blocked_lvl, ctx.engine.subtract(1.0, gate_blocked_lvl), ctx.rlk))
    dL_dgate_i_blocked = broadcast_full_to_blocks(ctx, dL_dgate_i, n_features, block_size)
    prod_t_blocked = ctx.engine.multiply(dL_dgate_i_blocked, surrogate_blocked, ctx.rlk)
    prod_t_blocked = ctx.engine.multiply(prod_t_blocked, sample_mask_blocked, ctx.rlk)
    prod_t_blocked = ctx.engine.intt(prod_t_blocked)
    sum_t_blocked = block_local_sum(ctx, prod_t_blocked, block_size)
    sum_t_blocked = _lvl(ctx, sum_t_blocked)
    sum_t_packed = gather_block_tops_to_packed(ctx, sum_t_blocked, n_features, block_size, ctx.engine.slot_count)
    dec_packed_vec = np.real(ctx.engine.decrypt(sum_t_packed, ctx.sk))

    w_for_t = _lvl(ctx, w)
    dL_dt_packed = ctx.engine.multiply(sum_t_packed, w_for_t, ctx.rlk)
    dL_dt_packed = ctx.engine.multiply(dL_dt_packed, -STEEPNESS)
    dL_dt_packed = _lvl(ctx, dL_dt_packed)
    for j in range(n_features):
        dL_dt_j = _extract_weight_broadcast(ctx, dL_dt_packed, j)
        dval = np.real(ctx.engine.decrypt(dL_dt_j, ctx.sk))[0]
        print(
            f"  feature {j}: sum_t={dec_packed_vec[j]:.6f} (diff={dec_packed_vec[j]-sum_t_baseline[j]:.6f})  "
            f"dL_dt_j={dval:.6f} (diff={dval-dL_dt_baseline[j]:.6f})"
        )


if __name__ == "__main__":
    main()
