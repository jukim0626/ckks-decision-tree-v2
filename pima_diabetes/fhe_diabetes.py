"""DESILO FHE soft Decision Tree inference for binary diabetes prediction."""

from __future__ import annotations

from desilofhe import Engine
import numpy as np


def create_ckks_context(max_level: int = 28) -> dict:
    """DESILO FHE Engine과 key 묶음을 생성합니다."""
    engine = Engine(max_level=max_level)
    sk = engine.create_secret_key()
    pk = engine.create_public_key(sk)
    rlk = engine.create_relinearization_key(sk)
    return {"engine": engine, "sk": sk, "pk": pk, "rlk": rlk}


def sigmoid_approx_enc(engine: Engine, rlk, enc_x):
    """5차 다항식으로 sigmoid를 근사합니다."""
    coeffs = [0.5, 0.209637, 0.0, -0.005402, 0.0, 0.000050]
    return engine.evaluate_polynomial(enc_x, coeffs, rlk)


def predict_ckks_binary(ctx: dict, sample: np.ndarray, structure: dict) -> int:
    """암호화된 feature로 binary soft Decision Tree 추론을 수행합니다."""
    engine: Engine = ctx["engine"]
    sk = ctx["sk"]
    pk = ctx["pk"]
    rlk = ctx["rlk"]

    children_left = structure["children_left"]
    children_right = structure["children_right"]
    feature = structure["feature"]
    threshold = structure["threshold"]
    value = structure["value"]

    n_nodes = len(children_left)
    leaf_id = -1

    enc_features = [
        engine.encrypt([float(sample[i])], pk)
        for i in range(len(sample))
    ]

    node_weights = [None] * n_nodes
    node_weights[0] = engine.encrypt([1.0], pk)

    class_scores = [
        engine.encrypt([0.0], pk),
        engine.encrypt([0.0], pk),
    ]

    for node_id in range(n_nodes):
        if node_weights[node_id] is None:
            continue

        if children_left[node_id] == leaf_id:
            node_value = np.array(value[node_id][0], dtype=float)
            leaf_prob = node_value / np.sum(node_value)
            class_scores[0] = engine.add(
                class_scores[0],
                engine.multiply(node_weights[node_id], float(leaf_prob[0])),
            )
            class_scores[1] = engine.add(
                class_scores[1],
                engine.multiply(node_weights[node_id], float(leaf_prob[1])),
            )
            continue

        feature_idx = feature[node_id]
        thresh = threshold[node_id]

        enc_diff = engine.subtract(enc_features[feature_idx], thresh)
        enc_right_prob = sigmoid_approx_enc(engine, rlk, enc_diff)
        enc_left_prob = engine.subtract(1.0, enc_right_prob)

        left_weight = engine.multiply(node_weights[node_id], enc_left_prob, rlk)
        right_weight = engine.multiply(node_weights[node_id], enc_right_prob, rlk)

        node_weights[children_left[node_id]] = left_weight
        node_weights[children_right[node_id]] = right_weight

    score_0 = engine.decrypt(class_scores[0], sk)[0]
    score_1 = engine.decrypt(class_scores[1], sk)[0]
    return int(score_1 >= score_0)
