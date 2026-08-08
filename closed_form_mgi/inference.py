"""학습된 ClosedFormMgiTreeModel로 임의의 encrypted dataset(test set이든 새 샘플이든)을
root->leaf로 라우팅하고, 최종 클래스 점수를 decrypt+argmax하는 추론 전용 모듈.
client_assisted/inference.py가 "decrypt는 최종 score에서 딱 한 번만" 원칙을 지키는 것과
동일한 경계를 따른다 - 여기서도 decrypt는 score_and_predict()에서만 일어난다.

thresholds/weights는 전부 train.py가 만든 모델에서 재사용하고, 여기서 새로 학습하는 건
없다 (원래 closed_form_mgi.py의 test_gate_from_thresholds가 하던 일과 동일)."""

from __future__ import annotations

import numpy as np

from ckks_tree import sigmoid_approx_enc
from closed_form_mgi.primitives import ensure_level
from closed_form_mgi.train import blended_gate_from_gates


def compute_gates_from_thresholds(ctx, dataset, thresholds: list):
    """train에서 구한 threshold(feature별 ciphertext)를 새 dataset의 feature에 그대로
    적용해서 gate만 계산 (재학습 없음)."""
    gates = []
    for j, threshold in enumerate(thresholds):
        enc_feature = ensure_level(ctx, dataset.enc_features[j])
        threshold = ensure_level(ctx, threshold)
        enc_diff = ctx.engine.subtract(enc_feature, threshold)
        gates.append(sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff))
    return gates


def route_dataset_through_model(ctx, model, dataset) -> list:
    """model.nodes를 pre-order로 순회하며 dataset을 root->leaf로 라우팅 (model을 학습한
    train_closed_form_mgi_tree의 순회 순서와 반드시 동일해야 함 - 둘 다 "현재 노드 처리 후
    왼쪽 서브트리 전체, 그다음 오른쪽 서브트리 전체" 순서의 pre-order).

    반환: leaf_weights(pre-order, model.leaf_counts와 같은 리프 순서로 정렬된, dataset의
    각 샘플이 그 leaf에 도달한 encrypted weight)."""
    leaf_weights: list = []
    node_iter = iter(model.nodes)

    def route(enc_weights, current_depth: int) -> None:
        if current_depth == model.depth:
            leaf_weights.append(enc_weights)
            return
        node = next(node_iter)
        enc_weights = ensure_level(ctx, enc_weights, min_level=16)
        gates = compute_gates_from_thresholds(ctx, dataset, node.thresholds)
        blended = blended_gate_from_gates(ctx, gates, node.weights)
        right = ctx.engine.multiply(enc_weights, blended, ctx.rlk)
        left = ctx.engine.multiply(enc_weights, ctx.engine.subtract(1.0, blended), ctx.rlk)
        route(left, current_depth + 1)
        route(right, current_depth + 1)

    root_weights = ctx.engine.encrypt([1.0] * dataset.n_samples, ctx.pk)
    route(root_weights, current_depth=0)
    return leaf_weights


def score_and_predict(ctx, model, leaf_weights: list, n_samples: int, n_classes: int) -> np.ndarray:
    """model.leaf_counts와 route_dataset_through_model()의 leaf_weights를 decrypt해서
    클래스 점수를 합산하고 argmax. 이 파이프라인에서 유일한 decrypt 지점."""

    def decrypt_scalar(ct) -> float:
        return float(ctx.engine.decrypt(ct, ctx.sk)[0].real)

    def decrypt_vector(ct, n: int) -> np.ndarray:
        return np.array([v.real for v in ctx.engine.decrypt(ct, ctx.sk)[:n]])

    leaf_counts_dec = [np.array([decrypt_scalar(c) for c in lc]) for lc in model.leaf_counts]
    leaf_weights_dec = [decrypt_vector(w, n_samples) for w in leaf_weights]

    scores = np.zeros((n_samples, n_classes))
    for w, counts in zip(leaf_weights_dec, leaf_counts_dec):
        scores += w[:, None] * counts[None, :]
    return np.argmax(scores, axis=1)
