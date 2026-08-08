"""교수님 요청: steepness - 정확도(평문 기준, 근사 X) 그래프, hard decision tree와 비교.

CKKS/polynomial 근사 없이 정확한 sigmoid(steepness * x)를 사용한 순수 numpy soft
decision tree. candidate grid, weighted-Gini split 선택, weighted-leaf 추론 로직은
client_assisted/의 production 알고리즘(candidates.py, client_ops.py,
verification.py)과 동일하게 맞췄다 - steepness 계수 자체만 다르다.

실행: python experiments/steepness_precision/steepness_precision_experiment.py
출력: steepness_precision_results.json, steepness_precision_plot.png (스크립트와 같은 폴더)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from client_assisted.candidates import make_small_public_threshold_grid
from client_assisted.dataset import load_scaled_dataset_subset

DATASETS = ["iris", "wine", "breast_cancer", "digits", "diabetes"]
DEPTH = 4
# digits(10-class, 8x8 픽셀)는 depth=4(leaf 16개)로도 hard tree 자체가 46.7%밖에 안 나와서
# (test_size=30 기준) 별도로 depth=8까지 올림 - depth=8이면 hard tree가 70%로 depth 무제한
# 트리(76.7%)에 근접함. 다른 데이터셋은 전부 DEPTH=4 그대로.
DEPTH_OVERRIDES = {"digits": 8}
# diabetes는 클래스 불균형(65:35)이 심해서 test_size=30이면 1샘플=3.3%p라 steepness별
# 경향이 노이즈에 묻힘 (steepness=1~2는 다수 클래스만 찍어도 66.7%가 나옴). test_size=150으로
# 늘려서 노이즈를 줄임 - train은 여전히 618개로 충분.
TEST_SIZE_OVERRIDES = {"diabetes": 150}
CANDIDATE_COUNT = 3
STEEPNESS_VALUES = list(range(1, 21))


def sigmoid(z: np.ndarray, steepness: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-steepness * z))


def weighted_gini(class_counts: np.ndarray) -> float:
    total = float(np.sum(class_counts))
    if total <= 1e-12:
        return 0.0
    probs = class_counts / total
    return total * (1.0 - float(np.sum(probs * probs)))


def train_soft_tree(
    X: np.ndarray,
    y_one_hot: np.ndarray,
    candidates: list[tuple[int, float]],
    depth: int,
    steepness: float,
) -> tuple[list[tuple[int, float]], list[np.ndarray]]:
    """client_assisted/training.py의 로직을 CKKS 없이 순수 numpy로 재현 (exact sigmoid)."""
    selections: list[tuple[int, float]] = []
    leaf_weights_list: list[np.ndarray] = []

    def train_node(node_weights: np.ndarray, current_depth: int) -> None:
        if current_depth == depth:
            leaf_weights_list.append(node_weights)
            return

        best_score = None
        best = None
        for feature_idx, threshold in candidates:
            diff = X[:, feature_idx] - threshold
            right_prob = sigmoid(diff, steepness)
            left_prob = 1.0 - right_prob
            right_w = node_weights * right_prob
            left_w = node_weights * left_prob
            left_counts = left_w @ y_one_hot
            right_counts = right_w @ y_one_hot
            score = weighted_gini(left_counts) + weighted_gini(right_counts)
            if best_score is None or score < best_score:
                best_score = score
                best = (feature_idx, threshold, left_w, right_w)

        feature_idx, threshold, left_w, right_w = best
        selections.append((feature_idx, threshold))
        train_node(left_w, current_depth + 1)
        train_node(right_w, current_depth + 1)

    train_node(np.ones(X.shape[0], dtype=float), current_depth=0)
    leaf_counts = [w @ y_one_hot for w in leaf_weights_list]
    return selections, leaf_counts


def predict_soft_tree(
    X: np.ndarray,
    selections: list[tuple[int, float]],
    leaf_counts: list[np.ndarray],
    depth: int,
    steepness: float,
) -> np.ndarray:
    leaf_weights: list[np.ndarray] = []
    idx = [0]

    def walk(node_weights: np.ndarray, current_depth: int) -> None:
        if current_depth == depth:
            leaf_weights.append(node_weights)
            return
        feature_idx, threshold = selections[idx[0]]
        idx[0] += 1
        diff = X[:, feature_idx] - threshold
        right_prob = sigmoid(diff, steepness)
        left_prob = 1.0 - right_prob
        walk(node_weights * left_prob, current_depth + 1)
        walk(node_weights * right_prob, current_depth + 1)

    walk(np.ones(X.shape[0], dtype=float), current_depth=0)
    scores = np.zeros((X.shape[0], leaf_counts[0].shape[0]), dtype=float)
    for w, counts in zip(leaf_weights, leaf_counts):
        scores += w[:, None] * counts[None, :]
    return np.argmax(scores, axis=1)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out


def main() -> None:
    results: dict[str, dict] = {}

    for dataset_name in DATASETS:
        depth = DEPTH_OVERRIDES.get(dataset_name, DEPTH)
        test_size = TEST_SIZE_OVERRIDES.get(dataset_name, 30)
        X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(
            dataset_name, test_size=test_size
        )
        n_classes = len(class_names)
        y_train_oh = one_hot(y_train, n_classes)
        candidates = [
            (feature_idx, threshold)
            for feature_idx in range(X_train.shape[1])
            for threshold in make_small_public_threshold_grid(CANDIDATE_COUNT)
        ]

        soft_accuracies = []
        for steepness in STEEPNESS_VALUES:
            selections, leaf_counts = train_soft_tree(
                X_train, y_train_oh, candidates, depth, float(steepness)
            )
            preds = predict_soft_tree(
                X_test, selections, leaf_counts, depth, float(steepness)
            )
            acc = float(np.mean(preds == y_test))
            soft_accuracies.append(acc)
            print(f"{dataset_name:14s} steepness={steepness:2d}  soft_acc={acc:.4f}")

        hard_clf = DecisionTreeClassifier(
            max_depth=depth, criterion="gini", random_state=42
        )
        hard_clf.fit(X_train, y_train)
        hard_acc = float(hard_clf.score(X_test, y_test))
        print(f"{dataset_name:14s} hard_tree_acc={hard_acc:.4f}")

        results[dataset_name] = {
            "depth": depth,
            "test_size": test_size,
            "steepness_values": STEEPNESS_VALUES,
            "soft_accuracies": soft_accuracies,
            "hard_accuracy": hard_acc,
        }

    output_path = Path(__file__).parent / "steepness_precision_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "depth": DEPTH,
                "depth_overrides": DEPTH_OVERRIDES,
                "test_size_overrides": TEST_SIZE_OVERRIDES,
                "candidate_count": CANDIDATE_COUNT,
                "sigmoid": "exact (no polynomial approximation)",
                "normalization": "MinMaxScaler(-1, 1)",
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nsaved -> {output_path}")


if __name__ == "__main__":
    main()
