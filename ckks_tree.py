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
    """암호화된 상태에서 n개 클래스 점수의 argmax를 계산.

    max(a, b) = 1/2(a+b) + 1/2*sign(a-b) 공식에 기반한 pairwise 비교를 사용.
    step(x) = (1 + sign(x)) / 2 ≈ sigmoid(COMPARE_SCALE * x) 로 근사.

    각 클래스 indicator:
      ind_i = prod_{j != i} step(s_i - s_j)
    즉, class i가 나머지 모두를 이길 때만 ~1에 수렴.

    최적화: step(s_j, s_i) = 1 - step(s_i, s_j) 이므로 (i < j)만 계산해서 캐시.
    """
    n_classes = len(class_scores)
    # None 클래스는 0으로 채움
    s = [
        cs if cs is not None else engine.encrypt([0.0], pk)
        for cs in class_scores
    ]

    def step(enc_a, enc_b):
        """step(a - b) = sigmoid(COMPARE_SCALE * (a - b)): a>b면 ~1, a<b면 ~0."""
        diff = engine.subtract(enc_a, enc_b)                   # level 소모 없음
        scaled = engine.multiply(diff, COMPARE_SCALE)          # level -1
        return sigmoid_approx_enc(engine, rlk, scaled)         # level -4

    # step(s[i], s[j])를 i < j 조합에 대해서만 계산 (sigmoid 호출 절반으로 축소)
    step_cache = {}
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            step_cache[(i, j)] = step(s[i], s[j])

    # 각 클래스 i의 indicator = j != i 에 대한 step(s[i], s[j]) 들의 곱
    indicators = []
    for i in range(n_classes):
        ind = None
        for j in range(n_classes):
            if i == j:
                continue
            if i < j:
                factor = step_cache[(i, j)]
            else:
                # step(s[i], s[j]) = 1 - step(s[j], s[i])
                factor = engine.subtract(1.0, step_cache[(j, i)])
            if ind is None:
                ind = factor
            else:
                ind = engine.multiply(ind, factor, rlk)        # CT×CT, level -1
        indicators.append(ind)

    # indicator 복호화 후 argmax
    decrypted = [engine.decrypt(ind, sk)[0] for ind in indicators]
    return int(np.argmax(decrypted))


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

    # 클래스 개수는 root value의 길이에서 자동 추론
    # value[node_id] shape: [[c0, c1, ..., c_{n-1}]]
    n_classes = len(value[0][0])
    class_scores = [None] * n_classes

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
