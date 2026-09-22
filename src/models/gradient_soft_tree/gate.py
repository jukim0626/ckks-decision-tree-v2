"""gradient_soft_tree(baseline)의 axis-aligned attention-blend gate 계산.

각 internal node의 gate = sum_j softmax(alpha)_j * sigmoid(steepness*(feature_j - threshold_j))
- feature마다 개별 sigmoid를 만들고 학습 가능한 softmax attention(alpha)으로 blend한다.

packed/는 이 함수를 그대로 안 쓴다 - 레이아웃 자체가 달라서(feature-axis SIMD block
packing) 이 형태로 재사용할 수 없다(자기 파일 안에 `_node_gate_packed`로 따로 있음).

2026-09-22: 예전엔 local_loss도 이 함수를 그대로 공유해서 썼으나(2026-09-09 리팩터 전에는
baseline과 local_loss 두 파일에 그대로 복붙돼 있었음) local_loss 계보 자체가 제거되면서
지금은 baseline만 쓴다."""

from __future__ import annotations

from core.ckks_engine import ensure_level
from core.encrypted_ops.slot_packing import extract_weight_broadcast
from core.approximation.sigmoid import sigmoid_approx_enc
from core.encrypted_ops.softmax import packed_softmax


def compute_axis_aligned_gate(
    ctx, dataset, alpha_i, threshold_i: list, n_features: int, n_pow2_f: int, min_level: int = 5,
):
    """노드 하나의 gate를 계산. 반환: (gate, gate_terms(feature별 sigmoid, backward에서
    재사용), w(softmax(alpha), backward에서 재사용))."""
    w = packed_softmax(ctx, alpha_i, n_features, n_pow2_f, min_level=min_level)

    gate_terms = []
    gate = None
    for j in range(n_features):
        enc_feature = ensure_level(ctx, dataset.enc_features[j], min_level=min_level)
        threshold_j = ensure_level(ctx, threshold_i[j], min_level=min_level)
        enc_diff = ctx.engine.subtract(enc_feature, threshold_j)
        # degree=15 sigmoid poly가 레벨을 정확히 5 소모한다 - poly 진입 전 별도로 더 높은
        # min_level을 보장해서 poly 이후에도 실제 여유가 남게 한다(2026-08-26 level_probe2.py 실측).
        enc_diff = ensure_level(ctx, enc_diff, min_level=12)
        gate_j = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff)
        gate_terms.append(gate_j)
        w_j = extract_weight_broadcast(ctx, w, j)
        piece = ctx.engine.multiply(w_j, gate_j, ctx.rlk)
        gate = piece if gate is None else ctx.engine.add(gate, piece)
    gate = ensure_level(ctx, gate, min_level=min_level)
    return gate, gate_terms, w
