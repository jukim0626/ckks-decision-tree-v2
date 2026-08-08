"""4가지 split 결정 방법을 plaintext로 비교: 정확도 + FHE 비용 proxy(sigmoid/reciprocal 호출 수).

방법:
  static_grid    : 지금 soft_mgi_plaintext.py의 baseline (public grid, K=n_feat*3, range norm)
  closed_form    : A안 - threshold = feature의 (node-)weighted mean, K=n_feat (grid 없음)
  dynamic_grid   : D안 - threshold grid = {mean-sigma, mean, mean+sigma} (데이터 기반, K=n_feat*3)
  gradient_refine: B안 - closed_form 초기값에서 analytic gradient로 3 step 정제, K=n_feat

비용 proxy: 실제 CKKS 연산 횟수를 이 프로젝트에서 이미 쓰는 단위(sigmoid_calls,
reciprocal_calls - 각각 desilofhe 다항식 평가/Newton-Raphson 반복 하나에 대응)로 센다.
sqrt는 아직 이 프로젝트에 구현된 적 없는 새 primitive라 별도로 표시.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from client_assisted.candidates import make_public_grid_candidates, make_small_public_threshold_grid
from client_assisted.dataset import load_scaled_dataset_subset, one_hot_encode

STEEPNESS = 8.0
BETA = 30.0
RECIPROCAL_ITERATIONS = 30  # soft_mgi.py 실측값과 동일 스케일로 비용 계산
GRADIENT_STEPS = 3
GRADIENT_LR = 1.5


class Cost:
    """FHE 연산 횟수 카운터 (plaintext 실행 중 함께 누적)."""

    def __init__(self):
        self.sigmoid_calls = 0
        self.reciprocal_calls = 0  # 1회 = Newton-Raphson RECIPROCAL_ITERATIONS번 반복
        self.sqrt_calls = 0  # 아직 이 프로젝트에 없는 primitive

    def as_dict(self):
        return {
            "acc": None,
            "sigmoid_calls": self.sigmoid_calls,
            "reciprocal_calls": self.reciprocal_calls,
            "reciprocal_equiv_multiplies": self.reciprocal_calls * RECIPROCAL_ITERATIONS,
            "sqrt_calls": self.sqrt_calls,
        }


def gate_fn(x_col, threshold, cost: Cost):
    cost.sigmoid_calls += 1
    return 1.0 / (1.0 + np.exp(-STEEPNESS * (x_col - threshold)))


def weighted_reciprocal_mean(x_col, weight, cost: Cost) -> float:
    """weighted mean = sum(weight*x)/sum(weight). division은 reciprocal 1회로 카운트."""
    cost.reciprocal_calls += 1
    total_w = weight.sum()
    if total_w <= 1e-9:
        return 0.0
    return float((weight * x_col).sum() / total_w)


def mgi(counts: np.ndarray) -> float:
    total = counts.sum()
    return float(total**2 - (counts**2).sum())


def build_candidate_pairs(method_name, X, y_onehot, node_weight, n_features, static_candidates, cost: Cost):
    """method_name에 따라 (feature_idx, threshold) 쌍 리스트를 만든다."""
    if method_name == "static_grid":
        return [(c.feature_idx, c.threshold) for c in static_candidates]

    if method_name == "closed_form":
        return [(j, weighted_reciprocal_mean(X[:, j], node_weight, cost)) for j in range(n_features)]

    if method_name == "dynamic_grid":
        pairs = []
        for j in range(n_features):
            mean = weighted_reciprocal_mean(X[:, j], node_weight, cost)
            mean_sq = weighted_reciprocal_mean(X[:, j] ** 2, node_weight, cost)
            var = max(mean_sq - mean**2, 1e-9)
            cost.sqrt_calls += 1
            sigma = float(np.sqrt(var))
            pairs.extend([(j, mean - sigma), (j, mean), (j, mean + sigma)])
        return pairs

    if method_name == "gradient_refine":
        score_normalizer = float(X.shape[0] ** 2)
        pairs = []
        for j in range(n_features):
            t = weighted_reciprocal_mean(X[:, j], node_weight, cost)
            for _ in range(GRADIENT_STEPS):
                rp = gate_fn(X[:, j], t, cost)
                dgate_dt = -STEEPNESS * rp * (1 - rp)
                lp, dlp_dt = 1 - rp, -dgate_dt
                left_c = ((node_weight * lp)[:, None] * y_onehot).sum(axis=0)
                right_c = ((node_weight * rp)[:, None] * y_onehot).sum(axis=0)
                dleft_c = ((node_weight * dlp_dt)[:, None] * y_onehot).sum(axis=0)
                dright_c = ((node_weight * dgate_dt)[:, None] * y_onehot).sum(axis=0)

                def mgi_grad(counts, dcounts):
                    total, dtotal = counts.sum(), dcounts.sum()
                    return 2 * total * dtotal - 2 * (counts * dcounts).sum()

                grad = mgi_grad(left_c, dleft_c) + mgi_grad(right_c, dright_c)
                t = t - GRADIENT_LR * (grad / score_normalizer)
            pairs.append((j, t))
        return pairs

    raise ValueError(method_name)


def score_and_blend(X, y_onehot, pairs, node_weight, cost: Cost):
    gates, left_list, right_list, scores = [], [], [], []
    for feat_idx, t in pairs:
        rp = gate_fn(X[:, feat_idx], t, cost)
        lp = 1 - rp
        left_c = ((node_weight * lp)[:, None] * y_onehot).sum(axis=0)
        right_c = ((node_weight * rp)[:, None] * y_onehot).sum(axis=0)
        gates.append(rp)
        left_list.append(left_c)
        right_list.append(right_c)
        scores.append(mgi(left_c) + mgi(right_c))
    scores = np.array(scores)
    lo, hi = scores.min(), scores.max()
    norm = (scores - lo) / max(hi - lo, 1e-9)
    cost.reciprocal_calls += 1  # softmax 정규화(가중치 합으로 나누기) 1회
    w = np.exp(-BETA * norm)
    w = w / w.sum()
    blended_gate = sum(w[i] * gates[i] for i in range(len(pairs)))
    return blended_gate, w


def train_and_eval(method_name, X_train, y_train_oh, X_test, y_test, static_candidates, n_features, n_classes, depth):
    cost = Cost()

    def build(node_weight, current_depth):
        if current_depth == depth:
            return ("leaf", (node_weight[:, None] * y_train_oh).sum(axis=0))
        pairs = build_candidate_pairs(method_name, X_train, y_train_oh, node_weight, n_features, static_candidates, cost)
        blended_gate, w = score_and_blend(X_train, y_train_oh, pairs, node_weight, cost)
        left = build(node_weight * (1 - blended_gate), current_depth + 1)
        right = build(node_weight * blended_gate, current_depth + 1)
        return ("split", pairs, w, left, right)

    root = build(np.ones(X_train.shape[0]), 0)

    def infer(node, X, node_weight, out_scores):
        if node[0] == "leaf":
            out_scores += node_weight[:, None] * node[1][None, :]
            return
        _, pairs, w, left, right = node
        gates = [gate_fn(X[:, feat_idx], t, cost) for feat_idx, t in pairs]
        blended_gate = sum(w[i] * gates[i] for i in range(len(pairs)))
        infer(left, X, node_weight * (1 - blended_gate), out_scores)
        infer(right, X, node_weight * blended_gate, out_scores)

    out_scores = np.zeros((X_test.shape[0], n_classes))
    infer(root, X_test, np.ones(X_test.shape[0]), out_scores)
    pred = np.argmax(out_scores, axis=1)
    acc = float((pred == y_test).mean())
    return acc, cost


METHODS = ["static_grid", "closed_form", "dynamic_grid", "gradient_refine"]
DATASETS = ["iris", "wine", "breast_cancer"]
DEPTHS = [1, 2, 3]
CANDIDATE_COUNT = 3

results = []
for dataset_name in DATASETS:
    X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(dataset_name, test_size=30)
    n_classes = len(class_names)
    n_features = X_train.shape[1]
    y_train_oh = one_hot_encode(y_train, n_classes=n_classes)
    candidates = make_public_grid_candidates(
        n_features=n_features, thresholds=make_small_public_threshold_grid(candidate_count=CANDIDATE_COUNT)
    )

    for depth in DEPTHS:
        sk_tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=0)
        sk_tree.fit(X_train, y_train)
        sk_acc = float((sk_tree.predict(X_test) == y_test).mean())
        print(f"\n=== {dataset_name} depth={depth} (sklearn={sk_acc:.3f}, n_features={n_features}) ===")
        row = {"dataset": dataset_name, "depth": depth, "n_features": n_features, "sklearn_acc": sk_acc}
        for method_name in METHODS:
            acc, cost = train_and_eval(
                method_name, X_train, y_train_oh, X_test, y_test, candidates, n_features, n_classes, depth
            )
            cd = cost.as_dict()
            cd["acc"] = acc
            row[method_name] = cd
            print(
                f"  {method_name:16s} acc={acc:.3f}  sigmoid={cd['sigmoid_calls']:4d}  "
                f"reciprocal={cd['reciprocal_calls']:3d}(~{cd['reciprocal_equiv_multiplies']:5d} mult-equiv)  "
                f"sqrt={cd['sqrt_calls']:2d}"
            )
        results.append(row)

with open("split_methods_plaintext_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved -> split_methods_plaintext_results.json")
