"""depthN_ckks.py의 `_debug_validate`와 완전히 같은 패턴(단일 프로세스로 몇 epoch만 돌려
plaintext reference와 대조) - `forward_backward_update_N` 대신
`forward_backward_update_N_packed`를 호출하고, setup 시 1회만 필요한 packing
결과물(blocked_features/sample_mask_blocked/block_masks/block_size)을 만들어 넘긴다.

**GPU 전용**: bootstrap은 CPU mode에서 사실상 끝나지 않을 정도로 느리다는 걸 실측으로
확인했다(level_preset=17, 단일 bootstrap 호출이 120초 넘게 안 끝남) - depthN_ckks.py의
`_debug_validate`도 애초에 mode="gpu"를 하드코딩해둔 이유가 이것으로 보인다. 이 스크립트도
동일하게 GPU만 쓴다.

python -m experiments.gradient_soft_tree.packed.debug_validate_packed <dataset> <depth> <epochs> <lr> [level_preset]
예: python -m experiments.gradient_soft_tree.packed.debug_validate_packed iris 1 3 2.0 17
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from client_assisted.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from closed_form_mgi.primitives import create_bootstrap_context  # noqa: E402
from experiments.gradient_soft_tree.depthN_ckks import decrypt_params_N, init_encrypted_params_N  # noqa: E402
from experiments.gradient_soft_tree.depthN_reference import train_depthN  # noqa: E402
from experiments.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    assert_layout_fits,
    broadcast_full_to_blocks,
    build_block_masks,
    compute_block_size,
    pack_dataset_features_blocked,
)
from experiments.gradient_soft_tree.packed.tree_ops_packed import forward_backward_update_N_packed  # noqa: E402


def _debug_validate_packed(dataset_name: str, depth: int, n_epochs: int, lr: float, seed: int = 0, level_preset: int | None = None):
    X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name)
    n_features = X_train.shape[1]
    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    y_train_oh = one_hot_encode(y_train, n_classes)

    print(
        f"[setup] dataset={dataset_name} depth={depth} n_features={n_features} n_classes={n_classes} "
        f"level_preset={level_preset}"
    )
    ctx = create_bootstrap_context(mode="gpu", level_preset=level_preset)
    dataset = encrypt_dataset(ctx, X_train, y_train_oh)
    sample_mask = ctx.engine.encrypt([1.0] * dataset.n_samples, ctx.pk)

    # --- packing 관련 setup: 데이터셋/epoch과 무관, 여기서 1회만 계산 ---
    block_size = compute_block_size(dataset.n_samples)
    assert_layout_fits(n_features, block_size, ctx.engine.slot_count)
    block_masks = build_block_masks(n_features, block_size, ctx.engine.slot_count)
    blocked_features = pack_dataset_features_blocked(ctx, dataset.enc_features, block_size)
    sample_mask_blocked = broadcast_full_to_blocks(ctx, sample_mask, n_features, block_size)
    print(f"[setup] block_size={block_size} (n_features*block_size={n_features * block_size} / slot_count={ctx.engine.slot_count})")

    params = init_encrypted_params_N(ctx, n_features, n_classes, depth, seed=seed, slot_count=ctx.engine.slot_count)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        params = forward_backward_update_N_packed(
            ctx, dataset, params, sample_mask, blocked_features, sample_mask_blocked,
            block_masks, block_size, n_features, n_classes, depth, lr=lr,
        )
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
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    lr_ = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    lp = int(sys.argv[5]) if len(sys.argv) > 5 else 17
    _debug_validate_packed(ds, d, n_ep, lr_, level_preset=lp)
