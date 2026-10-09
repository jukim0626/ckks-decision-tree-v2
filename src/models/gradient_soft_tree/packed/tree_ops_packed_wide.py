"""2026-09-30: `tree_ops_packed.py`(iris/wine/breast_cancer가 실제로 쓰는, 검증된 경로)는
이 파일 때문에 전혀 바뀌지 않는다 - soybean처럼 `n_features*block_size > slot_count`라
feature 전체를 ciphertext 하나에 못 담는 넓은 데이터셋을 위한 **완전히 별도의** forward/
backward다.

**핵심 아이디어**: feature 축을 `wide_features.plan_feature_groups`가 정한 그룹(각 그룹은
자기 ciphertext 하나에 들어갈 만큼만 feature를 담음)으로 나눈다. 그룹 하나짜리 케이스는
`tree_ops_packed.py`의 단일-ciphertext 로직과 수학적으로 완전히 동일(각 그룹 안에서
로컬 인덱스로 정확히 같은 packing/추출을 함) - 그룹이 여러 개면 그룹마다 독립적으로 계산한
뒤(서로 다른 feature라 완전히 독립) attention 가중치 `w`(전역 좌표계, feature grouping과
무관하게 원래도 하나의 작은 compact ciphertext)와 결합하기 위해 그룹 로컬<->전역 좌표계를
한 번씩 `rotate`로 옮긴다(`wide_features.extract_w_local`/`scatter_local_to_global`).

leaf class score/leaf backward/node-attention softmax 배치/reach-prob 전파는 feature 축과
전혀 무관하므로 `tree_ops_packed.py`의 해당 함수를 그대로 재사용한다(코드 중복 없음,
동일한 검증된 로직 공유)."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.approximation.sigmoid import sigmoid_approx_enc, STEEPNESS  # noqa: E402
from core.ckks_engine import ensure_level  # noqa: E402
from core.encrypted_ops.slot_packing import next_power_of_two, scatter_to_slot  # noqa: E402
from core.encrypted_ops.slot_packing import extract_weight_broadcast as _extract_weight_broadcast  # noqa: E402
from core.encrypted_ops.softmax import softmax_backward_packed  # noqa: E402
from models.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    block_local_sum,
    broadcast_full_to_blocks,
    extract_block_to_full,
    gather_block_tops_to_packed,
    pack_threshold_blocked,
)
from models.gradient_soft_tree.packed.wide_features import (  # noqa: E402
    extract_w_local,
    scatter_local_to_global,
)
from models.gradient_soft_tree.packed.tree_ops_packed import (  # noqa: E402
    _ensure_level,
    _batch_group_softmax,
    _leaf_class_scores,
    _propagate_reach_prob,
)


def _node_gate_packed_wide(ctx, w, threshold_i: list, blocked_features_groups: list, block_masks_local: list,
                            block_size: int, sample_mask, groups: list[tuple[int, int]]):
    """`tree_ops_packed._node_gate_packed`의 grouped 버전 - 그룹마다 threshold packing+
    sigmoid를 1번씩(그룹 개수만큼, 기존 단일-ciphertext 버전의 "전체 feature 1번"에 대응)
    부르고, 그 안에서 feature별 추출+attention 결합은 기존과 동일하다. `gate_blocked`
    하나 대신 그룹별 리스트를 반환해서 backward가 그룹별로 다시 쓸 수 있게 한다."""
    gate = None
    gate_blocked_groups = []
    for g_idx, (start, end) in enumerate(groups):
        n_local = end - start
        blocked_threshold_i_g = pack_threshold_blocked(ctx, threshold_i[start:end], block_masks_local[:n_local])
        enc_diff_blocked_g = ctx.engine.subtract(blocked_features_groups[g_idx], blocked_threshold_i_g)
        enc_diff_blocked_g = ensure_level(ctx, enc_diff_blocked_g, min_level=12)
        gate_blocked_g = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff_blocked_g)
        gate_blocked_groups.append(gate_blocked_g)

        w_local = extract_w_local(ctx, w, start)
        for local_j in range(n_local):
            gate_j = extract_block_to_full(ctx, gate_blocked_g, local_j, block_size, sample_mask)
            w_j = _extract_weight_broadcast(ctx, w_local, local_j)
            piece = ctx.engine.multiply(w_j, gate_j, ctx.rlk)
            gate = piece if gate is None else ctx.engine.add(gate, piece)
    gate = _ensure_level(ctx, gate)
    return gate, gate_blocked_groups, w


def predict_packed_wide(ctx, dataset, params: dict, sample_mask, blocked_features_groups: list,
                         block_masks_local: list, block_size: int, n_features: int, n_classes: int,
                         depth: int, groups: list[tuple[int, int]]):
    """`tree_ops_packed.predict_packed`의 grouped 버전 - forward만, backward용 캐시 없음."""
    n_leaves = 1 << depth

    current_level_probs = [None]
    node_idx = 0
    for _level in range(depth):
        count = len(current_level_probs)
        ws_this_level = _batch_group_softmax(ctx, params["alpha"][node_idx:node_idx + count], n_features, ctx.engine.slot_count)
        next_level_probs = []
        for level_idx, parent_prob in enumerate(current_level_probs):
            i = node_idx
            gate, _gate_blocked_groups, _w = _node_gate_packed_wide(
                ctx, ws_this_level[level_idx], params["threshold"][i], blocked_features_groups,
                block_masks_local, block_size, sample_mask, groups,
            )
            left, right = _propagate_reach_prob(ctx, gate, parent_prob)
            next_level_probs.append(left)
            next_level_probs.append(right)
            node_idx += 1
            gc.collect()
        current_level_probs = next_level_probs
    leaf_probs = current_level_probs

    y_hat, _leafdist = _leaf_class_scores(ctx, params["leaf_logits"], leaf_probs, n_classes, n_leaves)
    return y_hat


def forward_backward_update_N_packed_wide(ctx, dataset, params: dict, sample_mask, blocked_features_groups: list,
                                           sample_mask_blocked_groups: list, block_masks_local: list, block_size: int,
                                           n_features: int, n_classes: int, depth: int, lr: float,
                                           groups: list[tuple[int, int]]):
    """`tree_ops_packed.forward_backward_update_N_packed`의 grouped 버전. 알고리즘/gradient
    공식은 완전히 동일 - threshold/attention backward의 "전체 feature 1번" 연산들
    (`broadcast_full_to_blocks`/`block_local_sum`/`gather_block_tops_to_packed`)만 그룹마다
    반복하고, 그룹 로컬 결과를 전역 좌표계로 되돌려(threshold는 애초에 전역 인덱스라 바로
    쓰고, alpha gradient만 그룹별로 나온 걸 `scatter_local_to_global`로 옮겨서 합산) 최종
    dL_dalpha_i를 만든다."""
    n_pow2_c = next_power_of_two(n_classes)
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth
    slot_count = ctx.engine.slot_count

    node_gate = [None] * n_internal
    node_gate_blocked_groups = [None] * n_internal
    node_w = [None] * n_internal
    node_parent_prob = [None] * n_internal

    current_level_probs = [None]
    node_idx = 0
    for _level in range(depth):
        count = len(current_level_probs)
        ws_this_level = _batch_group_softmax(ctx, params["alpha"][node_idx:node_idx + count], n_features, slot_count)
        next_level_probs = []
        for level_idx, parent_prob in enumerate(current_level_probs):
            i = node_idx
            gate, gate_blocked_groups, w = _node_gate_packed_wide(
                ctx, ws_this_level[level_idx], params["threshold"][i], blocked_features_groups,
                block_masks_local, block_size, sample_mask, groups,
            )
            node_gate[i], node_gate_blocked_groups[i], node_w[i], node_parent_prob[i] = gate, gate_blocked_groups, w, parent_prob

            left, right = _propagate_reach_prob(ctx, gate, parent_prob)
            next_level_probs.append(left)
            next_level_probs.append(right)
            node_idx += 1
            gc.collect()
        current_level_probs = next_level_probs
    leaf_probs = current_level_probs

    y_hat, leafdist = _leaf_class_scores(ctx, params["leaf_logits"], leaf_probs, n_classes, n_leaves)

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
            term = ctx.engine.multiply(term, sample_mask)
            term = ctx.engine.intt(term)
            s = _ensure_level(ctx, ctx.engine.sum(term, ctx.rotation_key))
            piece = scatter_to_slot(ctx, s, c)
            dL_ddist_packed = piece if dL_ddist_packed is None else ctx.engine.add(dL_ddist_packed, piece)

            ld_c = _extract_weight_broadcast(ctx, leafdist[l], c)
            g_piece = ctx.engine.multiply(dL_dyhat[c], ld_c, ctx.rlk)
            g_l = g_piece if g_l is None else ctx.engine.add(g_l, g_piece)
        dL_dleaf_logits[l] = softmax_backward_packed(ctx, leafdist[l], _ensure_level(ctx, dL_ddist_packed), n_classes, n_pow2_c)
        current_g[l] = _ensure_level(ctx, g_l)
        gc.collect()

    new_threshold = [None] * n_internal
    new_alpha = [None] * n_internal
    for level in reversed(range(depth)):
        start_node = (1 << level) - 1
        count = 1 << level
        next_g = [None] * count
        for idx in range(count):
            i = start_node + idx
            g_left, g_right = current_g[2 * idx], current_g[2 * idx + 1]
            gate, w, gate_blocked_groups, p_i = node_gate[i], node_w[i], node_gate_blocked_groups[i], node_parent_prob[i]

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

            dL_dgate_i_masked = ctx.engine.multiply(dL_dgate_i, sample_mask)

            new_thresh_i = [None] * n_features
            dL_dalpha_i = None
            for g_idx, (g_start, g_end) in enumerate(groups):
                n_local = g_end - g_start
                gate_blocked_g = _ensure_level(ctx, gate_blocked_groups[g_idx])
                surrogate_blocked_g = ctx.engine.multiply(gate_blocked_g, ctx.engine.subtract(1.0, gate_blocked_g), ctx.rlk)
                surrogate_blocked_g = _ensure_level(ctx, surrogate_blocked_g)

                dL_dgate_i_blocked_g = broadcast_full_to_blocks(ctx, dL_dgate_i_masked, n_local, block_size)
                gate_broadcast_blocked_g = broadcast_full_to_blocks(ctx, gate, n_local, block_size)

                # threshold gradient (이 그룹의 feature들만)
                prod_t_blocked_g = ctx.engine.multiply(dL_dgate_i_blocked_g, surrogate_blocked_g, ctx.rlk)
                prod_t_blocked_g = ctx.engine.multiply(prod_t_blocked_g, sample_mask_blocked_groups[g_idx])
                prod_t_blocked_g = ctx.engine.intt(prod_t_blocked_g)
                sum_t_blocked_g = block_local_sum(ctx, prod_t_blocked_g, block_size)
                sum_t_blocked_g = _ensure_level(ctx, sum_t_blocked_g)
                sum_t_packed_g = gather_block_tops_to_packed(ctx, sum_t_blocked_g, n_local, block_size, slot_count)
                w_local = extract_w_local(ctx, _ensure_level(ctx, w), g_start)
                dL_dt_packed_g = ctx.engine.multiply(sum_t_packed_g, w_local, ctx.rlk)
                dL_dt_packed_g = ctx.engine.multiply(dL_dt_packed_g, -STEEPNESS)
                dL_dt_packed_g = _ensure_level(ctx, dL_dt_packed_g)

                for local_j in range(n_local):
                    global_j = g_start + local_j
                    dL_dt_j = _extract_weight_broadcast(ctx, dL_dt_packed_g, local_j)
                    new_thresh_i[global_j] = _ensure_level(
                        ctx, ctx.engine.subtract(params["threshold"][i][global_j], ctx.engine.multiply(dL_dt_j, lr))
                    )

                # attention gradient (이 그룹의 기여분 - 전역 좌표계로 되돌려서 누적)
                term_a_blocked_g = ctx.engine.multiply(dL_dgate_i_blocked_g, ctx.engine.subtract(gate_blocked_g, gate_broadcast_blocked_g), ctx.rlk)
                term_a_blocked_g = ctx.engine.multiply(term_a_blocked_g, sample_mask_blocked_groups[g_idx])
                term_a_blocked_g = ctx.engine.intt(term_a_blocked_g)
                sum_a_blocked_g = block_local_sum(ctx, term_a_blocked_g, block_size)
                sum_a_blocked_g = _ensure_level(ctx, sum_a_blocked_g)
                sum_a_packed_g = gather_block_tops_to_packed(ctx, sum_a_blocked_g, n_local, block_size, slot_count)

                dL_dalpha_i_g_local = ctx.engine.multiply(w_local, sum_a_packed_g, ctx.rlk)
                dL_dalpha_i_g_global = scatter_local_to_global(ctx, dL_dalpha_i_g_local, g_start)
                dL_dalpha_i = dL_dalpha_i_g_global if dL_dalpha_i is None else ctx.engine.add(dL_dalpha_i, dL_dalpha_i_g_global)

            dL_dalpha_i = _ensure_level(ctx, dL_dalpha_i)
            new_alpha[i] = _ensure_level(ctx, ctx.engine.subtract(params["alpha"][i], ctx.engine.multiply(dL_dalpha_i, lr)))
            new_threshold[i] = new_thresh_i
            gc.collect()
        current_g = next_g
        gc.collect()

    new_leaf_logits = [
        _ensure_level(ctx, ctx.engine.subtract(params["leaf_logits"][l], ctx.engine.multiply(dL_dleaf_logits[l], lr)))
        for l in range(n_leaves)
    ]

    return {"alpha": new_alpha, "threshold": new_threshold, "leaf_logits": new_leaf_logits}
