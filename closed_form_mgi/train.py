"""closed-form threshold(grid 없음) + soft-MGI blend 학습 (CKKS). depth>=1 재귀 지원.

2026-08-06 plaintext 검증(closed_form_mgi/debug/bench_split_methods_plaintext.py)에서
public grid(feature*3 candidates)를 없애고 feature마다 그 노드의 (node_weight로 가중한)
평균값을 threshold로 직접 계산하는 게, 정확도는 비슷하거나 더 좋으면서 candidate 수가
1/3로 줄어(feature*3 -> feature) 비용도 1/3 수준으로 낮다는 걸 확인했다. 이 파일은 그
방식을 실제 CKKS로 포팅한 것 (학습 전용 - 추론/라우팅은 closed_form_mgi/inference.py).

**아직 안 푼 부분**: plaintext 검증에서 depth>=2는 softmax 정규화를 "그 노드 K개 후보
점수의 min/max 범위"로 해야 beta<=50 안에서 hard-tournament 수준 정확도가 나왔는데(고정
정규화 n_samples**2로는 안 됨), encrypted min/max(비교 없이)는 아직 못 만들었다
(LogSumExp 기반 soft-min을 plaintext로 검증은 했지만 log() 다항식 근사가 새 primitive라
CKKS로 아직 안 옮겼다). 이 파일은 우선 **고정 정규화(score_normalizer=n_samples**2)**만
쓴다 - depth=1은 이걸로 충분(고정/range 정규화 정확도 동일 확인됨), depth>=2는 정확도가
plaintext range-normalization 버전보다 낮을 것으로 예상됨(알려진 한계, 실측 필요).

**train/inference 분리 관련 메모리 주의**: 원래(분리 전) 코드는 노드 하나를 처리할 때마다
train weight와 test weight를 동시에 계산하고 바로 버렸다 (peak 메모리 억제 목적, OOM 이력
있음). 분리 후에는 학습이 끝날 때까지 전체 트리의 노드별 thresholds/weights를
`ClosedFormMgiTreeModel.nodes`에 들고 있어야 라우팅(inference.py)이 나중에 재사용할 수
있다 - depth가 커지면 peak 메모리가 O(depth)에서 O(2^depth)로 늘어난다. depth=1로 먼저
검증하고 depth를 올리면서 메모리 문제가 없는지 확인할 것.
"""

from __future__ import annotations

import gc
import time

from ckks_tree import sigmoid_approx_enc
from client_assisted.server_ops import encrypted_class_counts
from closed_form_mgi.model import ClosedFormMgiNode, ClosedFormMgiTreeModel
from closed_form_mgi.primitives import (
    encrypted_mgi_from_counts,
    encrypted_weighted_class_counts,
    ensure_level,
)
from closed_form_mgi.simd_argmin import next_power_of_two, scatter_to_slot
from closed_form_mgi.soft_mgi import _extract_weight_broadcast, soft_mgi_weights

MEAN_RECIPROCAL_ITERATIONS = 30


def encrypted_weighted_threshold(ctx, enc_feature, enc_node_weights, n_samples: int, iterations: int = MEAN_RECIPROCAL_ITERATIONS):
    """threshold = weighted_mean(feature) = sum(weight*x)/sum(weight).

    soft_mgi.soft_mgi_weights와 같은 bounded Newton-Raphson 패턴: y0=1/n_samples(공개
    상한 기반 안전 초기값), z=weighted_count*y(0~1로 수렴, bootstrap-safe), w=weighted_sum*y
    (feature 값이 [-1,1] 안이라 가중평균도 자연히 [-1,1] 안 - bootstrap-safe). 수렴하면
    w == 원하는 threshold.
    """
    weighted_feature = ctx.engine.multiply(enc_feature, enc_node_weights, ctx.rlk)
    weighted_feature = ctx.engine.intt(weighted_feature)
    weighted_sum = ctx.engine.sum(weighted_feature, ctx.rotation_key)

    node_weights_for_sum = ctx.engine.intt(enc_node_weights)
    weighted_count = ctx.engine.sum(node_weights_for_sum, ctx.rotation_key)

    y0 = 1.0 / n_samples
    z = ctx.engine.multiply(weighted_count, y0)
    w = ctx.engine.multiply(weighted_sum, y0)
    for _ in range(iterations):
        z = ensure_level(ctx, z)
        w = ensure_level(ctx, w)
        two_minus_z = ctx.engine.subtract(2.0, z)
        z_new = ctx.engine.multiply(z, two_minus_z, ctx.rlk)
        w = ctx.engine.multiply(w, two_minus_z, ctx.rlk)
        z = z_new
    return w


def blended_gate_from_gates(ctx, gates: list, weights):
    """feature별 gate(각 gate는 "오른쪽으로 갈 확률")를 soft-MGI weight로 가중합.
    학습 중(자기 자신의 train weight 전파)과 추론 라우팅(inference.py, 새 dataset에 적용) 둘
    다에서 재사용되는 공용 함수라 여기(train.py)에 둔다."""
    weights = ensure_level(ctx, weights, min_level=14)
    gate = None
    for i, g_i in enumerate(gates):
        g_i = ensure_level(ctx, g_i, min_level=14)
        w_i = _extract_weight_broadcast(ctx, weights, i)
        piece = ctx.engine.multiply(w_i, g_i, ctx.rlk)
        gate = piece if gate is None else ctx.engine.add(gate, piece)
    return ensure_level(ctx, gate, min_level=16)


def evaluate_closed_form_candidates_from_thresholds(ctx, dataset, enc_node_weights, thresholds: list, score_normalizer: float):
    """미리 계산된 threshold(feature별)로 gate + MGI 점수를 계산해 SIMD 슬롯에 packing."""
    n_features = dataset.n_features
    n_pow2 = next_power_of_two(n_features)
    gates = []
    packed_score = None
    for j in range(n_features):
        enc_feature = ensure_level(ctx, dataset.enc_features[j])
        threshold = ensure_level(ctx, thresholds[j])
        enc_diff = ctx.engine.subtract(enc_feature, threshold)
        gate = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff)
        gates.append(gate)

        weighted_gate = ctx.engine.multiply(gate, enc_node_weights, ctx.rlk)
        left_prob = ctx.engine.subtract(1.0, gate)
        weighted_left = ctx.engine.multiply(left_prob, enc_node_weights, ctx.rlk)
        left_counts = encrypted_weighted_class_counts(ctx, weighted_left, dataset.enc_labels)
        right_counts = encrypted_weighted_class_counts(ctx, weighted_gate, dataset.enc_labels)
        score = ctx.engine.add(
            encrypted_mgi_from_counts(ctx, left_counts),
            encrypted_mgi_from_counts(ctx, right_counts),
        )
        # score는 최대 n_samples**2(~수천~수만) 크기라 bootstrap-safe 범위(~2~5)를 훨씬
        # 넘는다. ensure_level(=필요시 bootstrap)을 걸기 *전에* 반드시 score_normalizer로
        # 나눠서 [0,1] 안으로 넣어야 한다 (soft_mgi.py의 "값 크기 2~5 넘으면 bootstrap이
        # 조용히 깨진다" 버그와 동일 원인).
        score = ctx.engine.multiply(score, 1.0 / score_normalizer)
        score = ensure_level(ctx, score, min_level=14)
        piece = scatter_to_slot(ctx, score, j)
        packed_score = piece if packed_score is None else ctx.engine.add(packed_score, piece)
        gc.collect()

    if n_pow2 > n_features:
        pad_mask = ctx.engine.encrypt([0.0] * n_features + [1.0] * (n_pow2 - n_features), ctx.pk)
        sentinel = ctx.engine.encrypt([1.0] * n_pow2, ctx.pk)  # 이미 정규화된 스케일이라 "최악"=1.0
        packed_score = ctx.engine.add(packed_score, ctx.engine.multiply(pad_mask, sentinel, ctx.rlk))

    packed_score = ensure_level(ctx, packed_score, min_level=16)
    return packed_score, gates, n_pow2, n_features


def train_closed_form_mgi_tree(
    ctx,
    train_dataset,
    depth: int,
    beta: float = 30.0,
    score_normalizer: float | None = None,
    verbose: bool = True,
) -> ClosedFormMgiTreeModel:
    """client decrypt 없이, grid 없는 closed-form threshold + soft-MGI blend로 depth짜리
    tree를 학습. 반환: ClosedFormMgiTreeModel (노드별 thresholds/weights + leaf_counts,
    전부 pre-order). 새 dataset(test든 임의 샘플이든)을 이 모델로 라우팅하려면
    closed_form_mgi/inference.py의 route_dataset_through_model()을 쓴다."""
    if score_normalizer is None:
        score_normalizer = float(train_dataset.n_samples**2)
    if depth < 1:
        raise ValueError("depth must be at least 1")

    total_nodes = (1 << depth) - 1
    leaf_train_weights: list = []
    nodes: list = []
    node_count = [0]
    root_train_weights = ctx.engine.encrypt([1.0] * train_dataset.n_samples, ctx.pk)

    def train_node(enc_train_weights, current_depth: int) -> None:
        if current_depth == depth:
            leaf_train_weights.append(enc_train_weights)
            return
        enc_train_weights = ensure_level(ctx, enc_train_weights, min_level=16)
        t0 = time.time()

        thresholds = [
            encrypted_weighted_threshold(ctx, train_dataset.enc_features[j], enc_train_weights, train_dataset.n_samples)
            for j in range(train_dataset.n_features)
        ]
        packed_score, train_gates, n_pow2, n_features = evaluate_closed_form_candidates_from_thresholds(
            ctx, train_dataset, enc_train_weights, thresholds, score_normalizer
        )
        # packed_score는 이미 evaluate_closed_form_candidates_from_thresholds 안에서
        # score_normalizer로 정규화됐으므로, 여기서는 1.0을 넘겨 이중 정규화를 막는다.
        weights = soft_mgi_weights(ctx, packed_score, n_features, n_pow2, 1.0, beta)
        train_blended_gate = blended_gate_from_gates(ctx, train_gates, weights)

        nodes.append(ClosedFormMgiNode(thresholds=thresholds, weights=weights))

        node_count[0] += 1
        elapsed = time.time() - t0
        if verbose:
            print(f"[closed-form train] node {node_count[0]}/{total_nodes} (depth {current_depth}) done | {elapsed:.1f}s", flush=True)

        right_train = ctx.engine.multiply(enc_train_weights, train_blended_gate, ctx.rlk)
        left_train = ctx.engine.multiply(enc_train_weights, ctx.engine.subtract(1.0, train_blended_gate), ctx.rlk)
        del enc_train_weights
        gc.collect()
        train_node(left_train, current_depth + 1)
        del left_train
        train_node(right_train, current_depth + 1)

    train_node(root_train_weights, current_depth=0)
    leaf_counts = [encrypted_class_counts(ctx, lw, train_dataset.enc_labels) for lw in leaf_train_weights]
    return ClosedFormMgiTreeModel(depth=depth, nodes=nodes, leaf_counts=leaf_counts)
