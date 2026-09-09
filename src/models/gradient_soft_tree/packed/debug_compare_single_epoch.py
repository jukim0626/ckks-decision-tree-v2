"""baseline forward_backward_update_N과 packed forward_backward_update_N_packed을 같은
초기 encrypted params/dataset에서 딱 1 epoch만 돌려 decrypt해서 직접 비교(plaintext
reference가 아니라 서로 비교) - packed 버전이 plaintext 대비 오차가 빠르게 커지는 문제의
원인이 "누적 오차"가 아니라 "1 epoch 안에서 이미 공식이 다른가"를 가려내기 위한 디버그
전용 스크립트."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from core.ckks_engine import create_bootstrap_context  # noqa: E402
from models.gradient_soft_tree.baseline.tree_ops import (  # noqa: E402
    decrypt_params_N,
    forward_backward_update_N,
    init_encrypted_params_N,
)
from models.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    assert_layout_fits,
    broadcast_full_to_blocks,
    build_block_masks,
    compute_block_size,
    pack_dataset_features_blocked,
)
from models.gradient_soft_tree.packed.tree_ops_packed import forward_backward_update_N_packed  # noqa: E402


def main():
    dataset_name, depth, lr, seed, level_preset = "iris", 1, 2.0, 0, 17
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

    # baseline과 packed에 "정확히 같은" 초기 파라미터를 넣기 위해 같은 seed로 두 번 초기화
    params_baseline = init_encrypted_params_N(ctx, n_features, n_classes, depth, seed=seed, slot_count=ctx.engine.slot_count)
    params_packed = init_encrypted_params_N(ctx, n_features, n_classes, depth, seed=seed, slot_count=ctx.engine.slot_count)

    new_baseline = forward_backward_update_N(ctx, dataset, params_baseline, sample_mask, n_features, n_classes, depth, lr=lr)
    new_packed = forward_backward_update_N_packed(
        ctx, dataset, params_packed, sample_mask, blocked_features, sample_mask_blocked,
        block_masks, block_size, n_features, n_classes, depth, lr=lr,
    )

    dec_baseline = decrypt_params_N(ctx, new_baseline, n_features, n_classes, depth)
    dec_packed = decrypt_params_N(ctx, new_packed, n_features, n_classes, depth)

    for key in ("alpha", "threshold", "leaf_logits"):
        a, b = dec_baseline[key], dec_packed[key]
        diff = np.abs(a - b)
        print(f"=== {key} ===")
        print("baseline:", np.round(a, 5))
        print("packed  :", np.round(b, 5))
        print(f"max_abs_diff={diff.max():.6f}  argmax={np.unravel_index(diff.argmax(), diff.shape)}")


if __name__ == "__main__":
    main()
