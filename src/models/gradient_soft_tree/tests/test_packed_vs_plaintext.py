"""packed(forward_backward_update_N_packed)가 depth=1~3에서 plaintext reference
(baseline.reference.train_depthN)와 실제로 정합하는지 확인하는 regression test - 몇
epoch만 돌려서 decrypt한 파라미터와 plaintext 값의 max abs diff가 허용 오차 안인지
`assert`한다(예전엔 print만 하고 그냥 종료해서 실패해도 exit 0이었다 - tests/grad_check.py를
tests/로 옮길 때와 같은 이유로 고침).

**GPU 전용**: bootstrap은 CPU mode에서 사실상 끝나지 않을 정도로 느리다는 걸 실측으로
확인했다(level_preset=17, 단일 bootstrap 호출이 120초 넘게 안 끝남). GPU가 없으면
`RuntimeError`로 건너뛴 사실을 명시하고 종료한다(조용히 통과 처리하지 않음).

python -m models.gradient_soft_tree.tests.test_packed_vs_plaintext <dataset> <depth> <epochs> <lr> [level_preset]
예: python -m models.gradient_soft_tree.tests.test_packed_vs_plaintext iris 1 3 2.0 17
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from core.ckks_engine import create_bootstrap_context  # noqa: E402
from models.gradient_soft_tree.baseline.tree_ops import decrypt_params_N, init_encrypted_params_N  # noqa: E402
from models.gradient_soft_tree.baseline.reference import train_depthN  # noqa: E402
from models.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    assert_layout_fits,
    broadcast_full_to_blocks,
    build_block_masks,
    compute_block_size,
    pack_dataset_features_blocked,
)
from models.gradient_soft_tree.packed.tree_ops_packed import forward_backward_update_N_packed  # noqa: E402


# depth=1~2, 소수 epoch 기준 그동안 실측된 오차가 0.002~0.01 범위였다(EXPERIMENT_LOG.md
# 여러 항목, 이번 세션의 iris depth=1 반복 실행에서도 0.0017~0.0021) - 그보다 훨씬 넉넉한
# 값으로 잡아서, 진짜 알고리즘 버그(자릿수가 다른 수준의 오차)만 잡아내고 정상적인 CKKS
# 노이즈 변동으로는 안 흔들리게 한다.
TOL = 0.05


def test_packed_vs_plaintext(dataset_name: str, depth: int, n_epochs: int, lr: float, seed: int = 0, level_preset: int | None = None):
    X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name)
    n_features = X_train.shape[1]
    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    y_train_oh = one_hot_encode(y_train, n_classes)

    print(
        f"[setup] dataset={dataset_name} depth={depth} n_features={n_features} n_classes={n_classes} "
        f"level_preset={level_preset}"
    )
    try:
        ctx = create_bootstrap_context(mode="gpu", level_preset=level_preset)
    except Exception as exc:
        # GPU/CUDA 관련 실패시 desilofhe가 정확히 어떤 예외 타입을 던지는지 GPU 없는
        # 환경에서 실측 확인은 못 했다 - 넓게 잡아서 "조용히 통과"가 아니라 명시적으로
        # 다시 던진다(원인은 그대로 보존).
        raise RuntimeError(
            "GPU 엔진 생성 실패 - GPU가 없거나 사용할 수 없는 상태로 보입니다. 이 test는 "
            "CPU mode를 지원하지 않습니다(bootstrap이 사실상 끝나지 않을 정도로 느림). "
            f"원인: {exc!r}"
        ) from exc
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
        status = "OK" if max_err < TOL else "FAIL"
        print(f"[{status}] epoch {epoch} elapsed={elapsed:.1f}s  max abs diff vs plaintext = {max_err:.5f} (TOL={TOL})")
        assert max_err < TOL, f"epoch {epoch}: max_err={max_err:.5f} >= TOL={TOL}"


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "iris"
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    lr_ = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    lp = int(sys.argv[5]) if len(sys.argv) > 5 else 17
    test_packed_vs_plaintext(ds, d, n_ep, lr_, level_preset=lp)
