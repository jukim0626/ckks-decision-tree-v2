"""depthN_ckks.forward_backward_update_N을 feature-axis SIMD packing으로 재구성한 버전.
baseline(experiments/gradient_soft_tree/depthN_ckks.py)은 절대 import/수정하지 않는다 -
이 파일은 그 파일의 로직을 복제한 뒤 forward의 per-feature sigmoid 루프와 backward의
per-feature threshold/attention reduction 루프만 block_ops.py의 새 프리미티브로 바꿨다.

**핵심 변경**: `for j in range(n_features): sigmoid_approx_enc(...)`(n_features번 호출,
각각 bootstrap 유발)를 "모든 feature의 (x_j-threshold_j)를 block 레이아웃 하나로 packing
-> sigmoid_approx_enc 1번"으로 축소. backward의 threshold/attention gradient 계산도
`ctx.engine.sum`(전체 32768폭, feature마다 개별 호출)을 `block_local_sum`(block_size폭,
1번 호출로 전체 feature 동시 처리)으로 축소. 나머지(leaf 계산, leaf backward, 노드 간
level 전파)는 baseline과 완전히 동일 - 자세한 설계 근거는
`/home/juhyun/.claude/plans/dreamy-meandering-snowglobe.md` 참고.

**scope 제한(계획서에 명시)**: `params["threshold"][i]`는 baseline과 동일하게
`list[Ciphertext]`(feature별 broadcast ciphertext) 포맷을 그대로 쓴다 - 매 호출마다
`pack_threshold_blocked`로 packing했다가, 갱신 후 `_extract_weight_broadcast`로 다시
feature별로 꺼낸다. 이러면 `init_encrypted_params_N`/`decrypt_params_N`(baseline 그대로
재사용)과 params dict 모양이 완전히 같아서 I/O 계약이 단순해진다 - threshold를 완전히
packed 포맷으로 옮기는 건 독립적인 후속 최적화로 미룬다."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.approximation.sigmoid import sigmoid_approx_enc  # noqa: E402
from core.ckks_engine import ensure_level  # noqa: E402
from core.encrypted_ops.slot_packing import next_power_of_two, scatter_to_slot  # noqa: E402
from core.encrypted_ops.slot_packing import extract_weight_broadcast as _extract_weight_broadcast  # noqa: E402
from models.gradient_soft_tree.baseline.depth1_ckks import (  # noqa: E402
    STEEPNESS,
    packed_softmax,
    softmax_backward_packed,
)
from models.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    block_local_sum,
    broadcast_full_to_blocks,
    extract_block_to_full,
    gather_block_tops_to_packed,
    pack_threshold_blocked,
)

# depthN_ckks.py의 _LOCAL_MIN_LEVEL과 동일 이유/값 - baseline과 정확히 비교 가능해야 하므로
# 그대로 맞춘다.
_LOCAL_MIN_LEVEL = 5


def _ensure_level(ctx, ct):
    return ensure_level(ctx, ct, min_level=_LOCAL_MIN_LEVEL)


def predict_packed(
    ctx,
    dataset,
    params: dict,
    sample_mask,
    blocked_features,
    block_masks: list,
    block_size: int,
    n_features: int,
    n_classes: int,
    depth: int,
):
    """학습된(암호화 상태 그대로인) params로 `dataset`(보통 test set)에 대해 **forward만**
    실행해서 클래스별 encrypted score(y_hat)를 계산한다 - `forward_backward_update_N_packed`의
    forward 절반(leaf 계산까지)만 떼어낸 것으로, 코드를 그대로 복붙했다(공유 헬퍼로 뽑으면
    이미 검증된 학습 경로를 건드릴 위험이 있어 이 파일의 다른 곳처럼 "복제 후 최소satisfying
    변경" 원칙을 따름). params(alpha/threshold/leaf_logits)는 절대 decrypt하지 않는다 -
    유일한 decrypt 지점은 이 함수가 반환하는 y_hat뿐이다(client_assisted/inference.py
    모듈 docstring의 "client가 아는 유일한 예외적 decrypt 지점" 원칙과 동일)."""
    n_pow2_f = next_power_of_two(n_features)
    n_pow2_c = next_power_of_two(n_classes)
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth

    current_level_probs = [None]
    node_idx = 0
    for _level in range(depth):
        next_level_probs = []
        for parent_prob in current_level_probs:
            i = node_idx
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
            gate = _ensure_level(ctx, gate)

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
            ld_c = _ensure_level(ctx, _extract_weight_broadcast(ctx, leafdist[l], c))
            term = ctx.engine.multiply(leaf_probs[l], ld_c, ctx.rlk)
            acc = term if acc is None else ctx.engine.add(acc, term)
        y_hat.append(_ensure_level(ctx, acc))
    return y_hat


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
    """baseline forward_backward_update_N과 시그니처가 거의 같되, setup 시 1회만 만들면
    되는 packing 결과물(blocked_features/sample_mask_blocked/block_masks/block_size)을
    추가로 받는다 - 이것들은 epoch마다 안 바뀌므로 setup_worker_packed.py가 한 번만 만들어
    넘긴다."""
    n_pow2_f = next_power_of_two(n_features)
    n_pow2_c = next_power_of_two(n_classes)
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth
    slot_count = ctx.engine.slot_count

    node_gate = [None] * n_internal
    node_gate_blocked = [None] * n_internal
    node_w = [None] * n_internal
    node_parent_prob = [None] * n_internal

    current_level_probs = [None]
    node_idx = 0
    for _level in range(depth):
        next_level_probs = []
        for parent_prob in current_level_probs:
            i = node_idx
            w = packed_softmax(ctx, params["alpha"][i], n_features, n_pow2_f)

            # --- 이 5줄이 baseline의 `for j in range(n_features): ...` 루프(n_features번의
            # sigmoid_approx_enc 호출, 각각 bootstrap 유발)를 대체한다 ---
            blocked_threshold_i = pack_threshold_blocked(ctx, params["threshold"][i], block_masks)
            enc_diff_blocked = ctx.engine.subtract(blocked_features, blocked_threshold_i)
            enc_diff_blocked = ensure_level(ctx, enc_diff_blocked, min_level=12)  # baseline과 동일 guard(depthN_ckks.py 주석 참고)
            gate_blocked = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff_blocked)

            gate = None
            for j in range(n_features):  # 여기부턴 rotation+mask만(bootstrap 없음)
                # **마스킹이 baseline과 달라도 필요한 이유**: baseline의 gate_j는 회전 없는
                # 단일 feature ciphertext라 n_samples 밖은 원래부터(encrypt 시점부터) 0으로
                # 깨끗하다. 반면 packed gate_j는 gate_blocked(폭 n_features*block_size 밖은
                # sigmoid(0)=0.5, 그 안도 다른 feature block의 회전-랩어라운드 잔재가 섞일 수
                # 있음)를 "회전"해서 얻으므로 애초에 안 깨끗하다 - 여기서 sample_mask로
                # 명시적으로 지우지 않으면 gate(및 이걸로 만드는 gate_broadcast_blocked)가
                # n_samples 밖에서 오염된 채로 backward까지 전파된다(2026-09 디버깅에서
                # 실측: 마스킹을 빼면 alpha/threshold가 baseline과 1 epoch만에 이미
                # 0.003~0.009 어긋남 - leaf_logits는 이 마스킹과 무관한 코드라 정상이었음).
                gate_j = extract_block_to_full(ctx, gate_blocked, j, block_size, sample_mask)
                w_j = _extract_weight_broadcast(ctx, w, j)
                piece = ctx.engine.multiply(w_j, gate_j, ctx.rlk)
                gate = piece if gate is None else ctx.engine.add(gate, piece)
            gate = _ensure_level(ctx, gate)

            node_gate[i], node_gate_blocked[i], node_w[i], node_parent_prob[i] = gate, gate_blocked, w, parent_prob

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

    # --- leaf 계산/backward는 feature 축과 무관 - baseline과 완전히 동일 ---
    leafdist = [packed_softmax(ctx, params["leaf_logits"][l], n_classes, n_pow2_c) for l in range(n_leaves)]

    y_hat = []
    for c in range(n_classes):
        acc = None
        for l in range(n_leaves):
            ld_c = _ensure_level(ctx, _extract_weight_broadcast(ctx, leafdist[l], c))
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

            ld_c = _extract_weight_broadcast(ctx, leafdist[l], c)
            g_piece = ctx.engine.multiply(dL_dyhat[c], ld_c, ctx.rlk)
            g_l = g_piece if g_l is None else ctx.engine.add(g_l, g_piece)
        dL_dleaf_logits[l] = softmax_backward_packed(ctx, leafdist[l], _ensure_level(ctx, dL_ddist_packed), n_classes, n_pow2_c)
        current_g[l] = _ensure_level(ctx, g_l)
        gc.collect()

    # --- threshold/attention backward: baseline의 per-feature `ctx.engine.sum`(feature마다
    # 전체 32768폭 reduction)을 block_local_sum(block_size폭, 1번 호출로 전체 feature 동시
    # 처리)으로 대체 ---
    new_threshold = [None] * n_internal
    new_alpha = [None] * n_internal
    for level in reversed(range(depth)):
        start = (1 << level) - 1
        count = 1 << level
        next_g = [None] * count
        for idx in range(count):
            i = start + idx
            g_left, g_right = current_g[2 * idx], current_g[2 * idx + 1]
            gate, w, gate_blocked, p_i = node_gate[i], node_w[i], node_gate_blocked[i], node_parent_prob[i]

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

            gate_blocked = _ensure_level(ctx, gate_blocked)
            surrogate_blocked = ctx.engine.multiply(gate_blocked, ctx.engine.subtract(1.0, gate_blocked), ctx.rlk)
            surrogate_blocked = _ensure_level(ctx, surrogate_blocked)

            # **핵심 마스킹**: dL_dgate_i는 n_samples 밖에서 깨끗한 0이 아니다 - forward의
            # `left = 1 - gate`가 gate=0인 자리(=n_samples 밖)에서 1.0이 되고(0을 상수에서
            # 빼는 거라 clean하지 않음), 이 "1"이 leaf_probs -> y_hat -> dL_dyhat -> g_l ->
            # current_g -> dL_dgate_i까지 그대로 전파되어 dL_dgate_i의 n_samples 밖 값이
            # 어떤 nonzero 상수가 된다(2026-09 디버깅에서 실측 확인: broadcast_full_to_blocks
            # 없이 dL_dgate_i 자체는 baseline과 거의 완벽히 일치(diff~1e-7)했는데, block마다
            # 복제하는 순간 이 "먼 영역"의 nonzero 값이 회전+합산을 통해 **모든 block의 실제
            # 데이터 영역에 똑같이 새어 들어와서** threshold/attention gradient가 baseline과
            # 어긋났다 - feature별 diff가 rotation 여부와 무관하게 전부 똑같았던 게 결정적
            # 단서). gate는 gate_j가 이미 extract_block_to_full에서 sample_mask로 마스킹되니
            # clean해서 이 마스킹이 필요 없다 - dL_dgate_i만 여기서 명시적으로 마스킹한다.
            dL_dgate_i_masked = ctx.engine.multiply(dL_dgate_i, sample_mask, ctx.rlk)
            dL_dgate_i_blocked = broadcast_full_to_blocks(ctx, dL_dgate_i_masked, n_features, block_size)
            gate_broadcast_blocked = broadcast_full_to_blocks(ctx, gate, n_features, block_size)

            # threshold gradient: 전체 feature를 한 번에
            prod_t_blocked = ctx.engine.multiply(dL_dgate_i_blocked, surrogate_blocked, ctx.rlk)
            prod_t_blocked = ctx.engine.multiply(prod_t_blocked, sample_mask_blocked, ctx.rlk)
            prod_t_blocked = ctx.engine.intt(prod_t_blocked)
            sum_t_blocked = block_local_sum(ctx, prod_t_blocked, block_size)
            sum_t_blocked = _ensure_level(ctx, sum_t_blocked)
            sum_t_packed = gather_block_tops_to_packed(ctx, sum_t_blocked, n_features, block_size, slot_count)
            w_for_t = _ensure_level(ctx, w)
            dL_dt_packed = ctx.engine.multiply(sum_t_packed, w_for_t, ctx.rlk)
            dL_dt_packed = ctx.engine.multiply(dL_dt_packed, -STEEPNESS)
            dL_dt_packed = _ensure_level(ctx, dL_dt_packed)

            new_thresh_i = []
            for j in range(n_features):  # rotation+mask만(bootstrap 없음) - baseline의 w_j 추출과 동급 비용
                dL_dt_j = _extract_weight_broadcast(ctx, dL_dt_packed, j)
                new_thresh_i.append(
                    _ensure_level(ctx, ctx.engine.subtract(params["threshold"][i][j], ctx.engine.multiply(dL_dt_j, lr)))
                )

            # attention gradient: 전체 feature를 한 번에
            term_a_blocked = ctx.engine.multiply(dL_dgate_i_blocked, ctx.engine.subtract(gate_blocked, gate_broadcast_blocked), ctx.rlk)
            term_a_blocked = ctx.engine.multiply(term_a_blocked, sample_mask_blocked, ctx.rlk)
            term_a_blocked = ctx.engine.intt(term_a_blocked)
            sum_a_blocked = block_local_sum(ctx, term_a_blocked, block_size)
            sum_a_blocked = _ensure_level(ctx, sum_a_blocked)
            sum_a_packed = gather_block_tops_to_packed(ctx, sum_a_blocked, n_features, block_size, slot_count)

            w_lvl = _ensure_level(ctx, w)
            dL_dalpha_i = ctx.engine.multiply(w_lvl, sum_a_packed, ctx.rlk)
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
