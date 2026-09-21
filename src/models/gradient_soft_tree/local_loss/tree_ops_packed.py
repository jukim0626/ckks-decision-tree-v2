"""local_loss/tree_ops.py(ReBoot local-loss, virtual-leaf axis packing 이미 적용됨)에
packed/tree_ops_packed.py의 feature-axis SIMD packing을 결합한 버전.

**동기 (2026-09-11)**: local_loss는 지금까지 gate 계산(forward의 feature별 sigmoid,
backward의 feature별 threshold/attention gradient)을 baseline과 똑같이 **unpacked**
(feature마다 개별 호출) 방식 그대로 썼다 - local_loss가 최적화한 건 자신이 새로 추가한
"레벨별 loss 계산"(virtual_leaf x class 축)뿐이었다. 그 결과 feature가 많은 데이터셋
(wine 13개, breast_cancer 30개)에서는 baseline이 예전에 겪었던 것과 똑같은 병목
(feature마다 bootstrap 유발 sigmoid/reduction)을 그대로 물려받는다 - packed/가 wine
depth=3을 ~100시간→3.78시간으로 줄인 바로 그 문제.

**이 파일의 역할**: gate 관련 부분(forward의 per-feature sigmoid 루프, backward의
threshold/attention gradient)을 packed/tree_ops_packed.py와 동일한 feature-axis block
packing으로 교체한다. local_loss 고유의 레벨별 loss 계산(virtual_leaf x class 축 packing)은
그대로 재사용 - 두 packing 축(feature-axis, virtual-leaf-axis)은 독립적이라 함께 쓸 수 있다.
(2026-09-21: 두 축이 내부적으로 쓰던 block SIMD primitive가 완전히 동일한 코드였음이
확인돼 `core.encrypted_ops.block_ops`로 통합됐다 - 이 파일은 이제 그 하나의 원본을
feature-axis/virtual-leaf-axis 양쪽에 그대로 재사용한다.)

baseline(`local_loss/tree_ops.py`)과 packed(`packed/tree_ops_packed.py`)는 절대 안
건드림 - 이 파일은 둘의 로직을 조합해서 새로 짠 것.

**breast_cancer(30 feature) 주의**: `packed/block_ops.py`의 feature-axis 레이아웃은
`n_features * block_size <= slot_count`(32768)를 요구하는데, breast_cancer 기본
train set(539개)은 block_size=2048이라 30*2048=61440으로 넘친다 -
`setup_worker.py --max-train 512`(또는 그 이하)로 block_size를 1024로 낮춰야
30*1024=30720으로 들어간다(`assert_layout_fits`가 이 조건을 강제로 체크)."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.ckks_engine import ensure_level  # noqa: E402
from core.approximation.sigmoid import sigmoid_approx_enc, STEEPNESS  # noqa: E402
from core.encrypted_ops.slot_packing import extract_weight_broadcast, next_power_of_two  # noqa: E402
from core.encrypted_ops.softmax import packed_softmax, softmax_backward_packed  # noqa: E402
from models.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    broadcast_full_to_blocks,
    extract_block_to_full,
    gather_block_tops_to_packed,
    pack_threshold_blocked,
)
from models.gradient_soft_tree.local_loss.tree_ops import (  # noqa: E402
    init_encrypted_params_N,
    decrypt_params_N,
)
# 2026-09-21 1차 리팩터: local_loss 고유의 virtual-leaf axis packing에 쓰던
# _block_size_for/_scatter_terms_to_blocks/_block_local_sum/_gather_block_tops(예전엔
# "tree_ops.py의 private helper라 import 대신 여기 복붙"했던 것)는 feature-axis용
# gather_block_tops_to_packed와 수학적으로 완전히 동일한 코드였다 - core.encrypted_ops.
# block_ops의 통합된 원본을 두 축(feature-axis/virtual-leaf-axis) 모두에 그대로 재사용한다
# (수치 동작 변경 없음).
from core.encrypted_ops.block_ops import (  # noqa: E402
    block_local_sum,
    compute_block_size,
    gather_block_tops,
    scatter_to_blocks,
)

_LOCAL_MIN_LEVEL = 5


def _ensure_level(ctx, ct):
    return ensure_level(ctx, ct, min_level=_LOCAL_MIN_LEVEL)


def forward_backward_update_N_packed(
    ctx,
    dataset,
    params: dict,
    sample_mask,
    blocked_features,
    sample_mask_blocked,
    block_masks: list,
    block_size: int,
    n_features: int,
    n_classes: int,
    depth: int,
    lr: float,
):
    """local_loss forward_backward_update_N(tree_ops.py) + feature-axis packing
    (packed/tree_ops_packed.py). 시그니처는 packed 쪽과 동일하게 blocked_features/
    sample_mask_blocked/block_masks/block_size를 추가로 받는다(setup 시 1회 계산,
    epoch마다 재사용)."""
    n_pow2_f = next_power_of_two(n_features)
    n_pow2_c = next_power_of_two(n_classes)
    n_samples = dataset.n_samples
    slot_count = ctx.engine.slot_count

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
        node_gate_blocked = [None] * count
        node_w = [None] * count
        node_parent_prob = [None] * count

        # ---- forward: packed/tree_ops_packed.py와 동일한 feature-axis block packing ----
        for idx, parent_prob in enumerate(current_level_probs):
            i = node_idx
            w = packed_softmax(ctx, params["alpha"][i], n_features, n_pow2_f)

            blocked_threshold_i = pack_threshold_blocked(ctx, params["threshold"][i], block_masks)
            enc_diff_blocked = ctx.engine.subtract(blocked_features, blocked_threshold_i)
            enc_diff_blocked = ensure_level(ctx, enc_diff_blocked, min_level=12)
            gate_blocked = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff_blocked)

            gate = None
            for j in range(n_features):
                gate_j = extract_block_to_full(ctx, gate_blocked, j, block_size, sample_mask)
                w_j = extract_weight_broadcast(ctx, w, j)
                piece = ctx.engine.multiply(w_j, gate_j, ctx.rlk)
                gate = piece if gate is None else ctx.engine.add(gate, piece)
            gate = _ensure_level(ctx, gate)

            node_gate[idx], node_gate_blocked[idx], node_w[idx], node_parent_prob[idx] = gate, gate_blocked, w, parent_prob

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

        # ---- 이 레벨의 local classifier loss - tree_ops.py(unpacked 버전)와 100% 동일 ----
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

        loss_block_size = compute_block_size(n_samples)
        n_blocks = n_virtual_leaves * n_classes
        assert n_blocks * loss_block_size <= slot_count, (
            f"virtual_leaf x class 조합({n_blocks}) x block_size({loss_block_size})가 slot_count를 초과"
        )

        terms = []
        for k in range(n_virtual_leaves):
            for c in range(n_classes):
                term = ctx.engine.multiply(dL_dyhat_level[c], current_level_probs[k], ctx.rlk)
                term = ctx.engine.multiply(term, sample_mask, ctx.rlk)
                terms.append(ctx.engine.intt(term))
        blocked = scatter_to_blocks(ctx, terms, loss_block_size)
        reduced = _ensure_level(ctx, block_local_sum(ctx, blocked, loss_block_size))
        tops = gather_block_tops(ctx, reduced, n_blocks, loss_block_size, slot_count)

        class_mask = np.zeros(slot_count)
        class_mask[:n_classes] = 1.0

        g_this_level = [None] * n_virtual_leaves
        local_logits_grad = [None] * n_virtual_leaves
        for k in range(n_virtual_leaves):
            shift = k * n_classes
            window = tops if shift == 0 else ctx.engine.rotate(tops, ctx.rotation_key, -shift)
            dL_ddist_packed = ctx.engine.multiply(window, class_mask)

            g_k = None
            for c in range(n_classes):
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

        # ---- 이 레벨 gate 파라미터의 backward - packed/tree_ops_packed.py와 동일한
        # feature-axis block reduction으로 threshold/attention gradient 계산 ----
        for idx in range(count):
            i = start + idx
            g_left, g_right = g_this_level[2 * idx], g_this_level[2 * idx + 1]
            gate, w, gate_blocked, p_i = node_gate[idx], node_w[idx], node_gate_blocked[idx], node_parent_prob[idx]

            diff_g = _ensure_level(ctx, ctx.engine.subtract(g_right, g_left))
            if p_i is None:
                dL_dgate_i = diff_g
            else:
                dL_dgate_i = _ensure_level(ctx, ctx.engine.multiply(p_i, diff_g, ctx.rlk))

            gate_blocked = _ensure_level(ctx, gate_blocked)
            surrogate_blocked = ctx.engine.multiply(gate_blocked, ctx.engine.subtract(1.0, gate_blocked), ctx.rlk)
            surrogate_blocked = _ensure_level(ctx, surrogate_blocked)

            dL_dgate_i_masked = ctx.engine.multiply(dL_dgate_i, sample_mask, ctx.rlk)
            dL_dgate_i_blocked = broadcast_full_to_blocks(ctx, dL_dgate_i_masked, n_features, block_size)
            gate_broadcast_blocked = broadcast_full_to_blocks(ctx, gate, n_features, block_size)

            prod_t_blocked = ctx.engine.multiply(dL_dgate_i_blocked, surrogate_blocked, ctx.rlk)
            prod_t_blocked = ctx.engine.multiply(prod_t_blocked, sample_mask_blocked, ctx.rlk)
            prod_t_blocked = ctx.engine.intt(prod_t_blocked)
            sum_t_blocked = _ensure_level(ctx, block_local_sum(ctx, prod_t_blocked, block_size))
            sum_t_packed = gather_block_tops_to_packed(ctx, sum_t_blocked, n_features, block_size, slot_count)
            w_for_t = _ensure_level(ctx, w)
            dL_dt_packed = ctx.engine.multiply(sum_t_packed, w_for_t, ctx.rlk)
            dL_dt_packed = _ensure_level(ctx, ctx.engine.multiply(dL_dt_packed, -STEEPNESS))

            new_thresh_i = [
                _ensure_level(
                    ctx,
                    ctx.engine.subtract(
                        params["threshold"][i][j],
                        ctx.engine.multiply(extract_weight_broadcast(ctx, dL_dt_packed, j), lr),
                    ),
                )
                for j in range(n_features)
            ]

            term_a_blocked = ctx.engine.multiply(dL_dgate_i_blocked, ctx.engine.subtract(gate_blocked, gate_broadcast_blocked), ctx.rlk)
            term_a_blocked = ctx.engine.multiply(term_a_blocked, sample_mask_blocked, ctx.rlk)
            term_a_blocked = ctx.engine.intt(term_a_blocked)
            sum_a_blocked = _ensure_level(ctx, block_local_sum(ctx, term_a_blocked, block_size))
            sum_a_packed = gather_block_tops_to_packed(ctx, sum_a_blocked, n_features, block_size, slot_count)

            w_lvl = _ensure_level(ctx, w)
            dL_dalpha_i = ctx.engine.multiply(w_lvl, sum_a_packed, ctx.rlk)
            new_alpha[i] = _ensure_level(ctx, ctx.engine.subtract(params["alpha"][i], ctx.engine.multiply(dL_dalpha_i, lr)))
            new_threshold[i] = new_thresh_i
            gc.collect()
        gc.collect()

    return {"alpha": new_alpha, "threshold": new_threshold, "local_logits": new_local_logits}


def _debug_validate(dataset_name: str, depth: int, n_epochs: int, lr: float, seed: int = 0, level_preset: int | None = None, max_train: int | None = None):
    """local_loss/tree_ops.py의 _debug_validate와 같은 목적 - 단일 프로세스로 몇 epoch만
    돌려 plaintext(local_loss/reference.py의 train_depthN_local_loss)와 대조."""
    import time

    from core.ckks_engine import create_bootstrap_context
    from core.data.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode
    from models.gradient_soft_tree.local_loss.reference import train_depthN_local_loss
    from models.gradient_soft_tree.packed.block_ops import (
        assert_layout_fits,
        build_block_masks,
        compute_block_size,
        pack_dataset_features_blocked,
    )

    X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name, max_train=max_train)
    n_features = X_train.shape[1]
    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    y_train_oh = one_hot_encode(y_train, n_classes)

    print(f"[setup] dataset={dataset_name} depth={depth} n_features={n_features} n_classes={n_classes} level_preset={level_preset} n_train={X_train.shape[0]}")
    ctx = create_bootstrap_context(mode="gpu", level_preset=level_preset)
    dataset = encrypt_dataset(ctx, X_train, y_train_oh)
    sample_mask = ctx.engine.encrypt([1.0] * dataset.n_samples, ctx.pk)

    block_size = compute_block_size(dataset.n_samples)
    assert_layout_fits(n_features, block_size, ctx.engine.slot_count)
    block_masks = build_block_masks(n_features, block_size, ctx.engine.slot_count)
    blocked_features = pack_dataset_features_blocked(ctx, dataset.enc_features, block_size)
    sample_mask_blocked = broadcast_full_to_blocks(ctx, sample_mask, n_features, block_size)
    print(f"[setup] block_size={block_size} n_features*block_size={n_features * block_size} slot_count={ctx.engine.slot_count}")

    params = init_encrypted_params_N(ctx, n_features, n_classes, depth, seed=seed, slot_count=ctx.engine.slot_count)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        params = forward_backward_update_N_packed(
            ctx, dataset, params, sample_mask, blocked_features, sample_mask_blocked,
            block_masks, block_size, n_features, n_classes, depth, lr=lr,
        )
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
    mt = int(sys.argv[6]) if len(sys.argv) > 6 else None
    _debug_validate(ds, d, n_ep, lr_, level_preset=lp, max_train=mt)
