import tenseal as ts
import numpy as np


def create_ckks_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=32768,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60]  # depth 6 배치
    )
    context.global_scale = 2 ** 40
    return context

# decision tree(if/else) 역할
def sigmoid_approx_enc(enc_x):

    # 3차 다항식 sigmoid 근사 0.5 + 0.197x - 0.004x^3
    x2 = enc_x * enc_x        # depth 1
    x3 = x2 * enc_x           # depth 2
    return enc_x * 0.197 + x3 * (-0.004) + 0.5


def predict_ckks(context, sample, structure):
    children_left = structure["children_left"]
    children_right = structure["children_right"]
    feature = structure["feature"]
    threshold = structure["threshold"]
    value = structure["value"]

    n_nodes = len(children_left)
    LEAF = -1

    # 특징 4개 각각 encrypt -> 이후 plaintext 접근 불가
    enc_features = [
        ts.ckks_vector(context, [float(sample[i])])
        for i in range(len(sample))
    ]

    # 노드 weight 초기화
    node_weights = [None] * n_nodes
    node_weights[0] = ts.ckks_vector(context, [1.0])

    # 클래스 점수 None으로 초기화
    class_scores = [None] * 3

    for node_id in range(n_nodes):
        if node_weights[node_id] is None:
            continue

        # 리프 노드
        if children_left[node_id] == LEAF:
            node_value = value[node_id][0]
            pred_class = int(np.argmax(node_value))
            # None이면 그냥 할당 아니면 누적
            if class_scores[pred_class] is None:
                class_scores[pred_class] = node_weights[node_id]
            else:
                class_scores[pred_class] = class_scores[pred_class] + node_weights[node_id]
            continue

        # 분기 노드
        feat_idx = feature[node_id]
        thresh = threshold[node_id]

        # x - threshold
        enc_diff = enc_features[feat_idx] + (-thresh)

        # sigmoid 근사 (depth 2 소모)
        enc_right_prob = sigmoid_approx_enc(enc_diff)

        # 1 - right_prob
        enc_left_prob = 1.0 + (-1) * enc_right_prob

        # weight 전파 (depth 1 소모)
        left_weight  = node_weights[node_id] * enc_left_prob
        right_weight = node_weights[node_id] * enc_right_prob

        node_weights[children_left[node_id]]  = left_weight
        node_weights[children_right[node_id]] = right_weight

    # decrypt (None인 클래스는 0.0 처리)
    scores = [
        cs.decrypt()[0] if cs is not None else 0.0
        for cs in class_scores
    ]
    return int(np.argmax(scores))