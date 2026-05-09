from desilofhe import Engine
import numpy as np

# 비교 sigmoid를 sharp하게 만들기 위한 스케일
# 예: 점수 차이 0.2 → sigmoid(0.2)≈0.55 (무딤) vs sigmoid(5*0.2)≈0.73 (더 명확)
COMPARE_SCALE = 5.0


def create_ckks_context() -> dict:
    """desilofhe Engine과 키 묶음을 생성해서 반환.

    max_level=15: tree traversal(~7 level) + argmax 비교(~5 level) 여유분 포함.
    """
    engine = Engine(max_level=15)
    sk = engine.create_secret_key()
    pk = engine.create_public_key(sk)
    rlk = engine.create_relinearization_key(sk)
    return {"engine": engine, "sk": sk, "pk": pk, "rlk": rlk}


def sigmoid_approx_enc(engine: Engine, rlk, enc_x):
    """5차 다항식으로 sigmoid를 근사 (0~1 범위 출력).

    sigmoid(x) ≈ 0.5 + 0.209637x - 0.005402x^3 + 0.000050x^5
    evaluate_polynomial은 coeffs[i]를 i차 계수로 받는다.
    """
    coeffs = [0.5, 0.209637, 0.0, -0.005402, 0.0, 0.000050]
    return engine.evaluate_polynomial(enc_x, coeffs, rlk)


def encrypted_argmax(engine: Engine, rlk, sk, pk, class_scores: list) -> int:
    """암호화된 상태에서 3개 클래스 점수의 argmax를 계산.

    max(a, b) = 1/2(a+b) + 1/2*sign(a-b) 공식에 기반한 pairwise 비교를 사용.
    step(x) = (1 + sign(x)) / 2 ≈ sigmoid(COMPARE_SCALE * x) 로 근사.

    각 클래스 indicator:
      ind_i = step(si - sj) * step(si - sk)   (j, k ≠ i)
    즉, class i가 나머지 두 클래스를 모두 이길 때만 ~1에 수렴.
    """
    # None 클래스는 0으로 채움
    s = [
        cs if cs is not None else engine.encrypt([0.0], pk)
        for cs in class_scores
    ]
    s0, s1, s2 = s[0], s[1], s[2]

    def step(enc_a, enc_b):
        """step(a - b) = sigmoid(COMPARE_SCALE * (a - b)): a>b면 ~1, a<b면 ~0."""
        diff = engine.subtract(enc_a, enc_b)                   # level 소모 없음
        scaled = engine.multiply(diff, COMPARE_SCALE)          # level -1
        return sigmoid_approx_enc(engine, rlk, scaled)         # level -4

    # pairwise step 비교 (3쌍)
    step_01 = step(s0, s1)                          # s0 > s1 면 ~1
    step_02 = step(s0, s2)                          # s0 > s2 면 ~1
    step_12 = step(s1, s2)                          # s1 > s2 면 ~1

    # 반대 방향: 1 - step = step(b - a)
    step_10 = engine.subtract(1.0, step_01)         # s1 > s0 면 ~1
    step_20 = engine.subtract(1.0, step_02)         # s2 > s0 면 ~1
    step_21 = engine.subtract(1.0, step_12)         # s2 > s1 면 ~1

    # 각 클래스 indicator: 두 step 결과를 곱함 (CT × CT, level -1)
    ind_0 = engine.multiply(step_01, step_02, rlk)  # s0이 s1, s2 모두 이길 때 ~1
    ind_1 = engine.multiply(step_10, step_12, rlk)  # s1이 s0, s2 모두 이길 때 ~1
    ind_2 = engine.multiply(step_20, step_21, rlk)  # s2가 s0, s1 모두 이길 때 ~1

    # indicator 복호화 후 argmax
    indicators = [
        engine.decrypt(ind_0, sk)[0],
        engine.decrypt(ind_1, sk)[0],
        engine.decrypt(ind_2, sk)[0],
    ]
    return int(np.argmax(indicators))


def predict_ckks(ctx: dict, sample, structure) -> int:
    """암호화된 상태에서 decision tree 추론을 수행."""
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
    LEAF = -1

    # 특징 4개 각각 encrypt
    enc_features = [
        engine.encrypt([float(sample[i])], pk)
        for i in range(len(sample))
    ]

    # 루트 노드 weight = 1.0
    node_weights = [None] * n_nodes
    node_weights[0] = engine.encrypt([1.0], pk)

    class_scores = [None] * 3

    for node_id in range(n_nodes):
        if node_weights[node_id] is None:
            continue

        # 리프 노드: 해당 클래스 점수에 weight 누적
        if children_left[node_id] == LEAF:
            node_value = value[node_id][0]
            pred_class = int(np.argmax(node_value))
            if class_scores[pred_class] is None:
                class_scores[pred_class] = node_weights[node_id]
            else:
                class_scores[pred_class] = engine.add(
                    class_scores[pred_class], node_weights[node_id]
                )
            continue

        # 분기 노드
        feat_idx = feature[node_id]
        thresh = threshold[node_id]

        # x - threshold
        enc_diff = engine.subtract(enc_features[feat_idx], thresh)

        # sigmoid 근사로 오른쪽 자식 확률 계산
        enc_right_prob = sigmoid_approx_enc(engine, rlk, enc_diff)

        # 1 - right_prob = 왼쪽 자식 확률
        enc_left_prob = engine.subtract(1.0, enc_right_prob)

        # weight 전파
        left_weight = engine.multiply(node_weights[node_id], enc_left_prob, rlk)
        right_weight = engine.multiply(node_weights[node_id], enc_right_prob, rlk)

        node_weights[children_left[node_id]] = left_weight
        node_weights[children_right[node_id]] = right_weight

    # 암호화된 상태에서 argmax 계산
    return encrypted_argmax(engine, rlk, sk, pk, class_scores)
