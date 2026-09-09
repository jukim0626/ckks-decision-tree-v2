"""gradient_soft_tree 계열이 공유하는 axis-aligned attention-blend gate 계산.

각 internal node의 gate = sum_j softmax(alpha)_j * sigmoid(steepness*(feature_j - threshold_j))
- feature마다 개별 sigmoid를 만들고 학습 가능한 softmax attention(alpha)으로 blend한다.
baseline과 local_loss가 이 계산을 완전히 동일하게 쓴다(2026-09-09 리팩터 전에는 두 파일에
그대로 복붙돼 있었음 - 하나를 고치면 다른 하나도 똑같이 고쳐야 하는 상태였다).

opt/, packed/는 이 함수를 그대로 안 쓴다: opt는 이 계산 자체를 여러 방식으로 변형해서
bootstrap 횟수를 줄이는 실험이 목적이고(TreeConfig 플래그로 계산 자체가 달라짐), packed는
레이아웃 자체가 달라서(feature-axis SIMD block packing) 이 형태로 재사용할 수 없다 -
"모델별로 달라지는 개념은 따로 빼지 않는다"는 원칙에 따라 이 둘은 자기 파일 안에 그대로 둔다."""

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
