"""Train and export a deep Decision Tree for Pima Indians Diabetes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from pima_diabetes.data import (
    FEATURE_NAMES,
    PreprocessStats,
    fit_preprocess_stats,
    load_raw_dataset,
    standardized_threshold_to_raw,
    transform_features,
)


@dataclass(frozen=True)
class TrainResult:
    """학습 결과와 추론에 필요한 데이터를 담는 객체."""

    clf: DecisionTreeClassifier
    structure: dict
    x_test_raw: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    stats: PreprocessStats
    rules_text: str


def extract_tree_structure(clf: DecisionTreeClassifier) -> dict:
    """sklearn DecisionTreeClassifier에서 tree 구조를 dict로 추출합니다."""
    tree = clf.tree_
    return {
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": tree.value.tolist(),
        "n_node_samples": tree.n_node_samples.tolist(),
        "impurity": tree.impurity.tolist(),
    }


def export_if_else_rules(
    structure: dict,
    stats: PreprocessStats,
    node_id: int = 0,
    indent: int = 0,
) -> str:
    """tree 전체를 사람이 읽을 수 있는 if/else 분기 문자열로 변환합니다."""
    children_left = structure["children_left"]
    children_right = structure["children_right"]
    feature = structure["feature"]
    threshold = structure["threshold"]
    value = structure["value"]
    n_node_samples = structure["n_node_samples"]
    impurity = structure["impurity"]

    prefix = "    " * indent
    leaf_id = -1

    if children_left[node_id] == leaf_id:
        class_counts = np.array(value[node_id][0], dtype=float)
        pred_class = int(np.argmax(class_counts))
        total = float(np.sum(class_counts))
        diabetes_prob = class_counts[1] / total if total else 0.0
        return (
            f"{prefix}return class={pred_class} "
            f"(diabetes_prob={diabetes_prob:.3f}, "
            f"samples={n_node_samples[node_id]}, "
            f"gini={impurity[node_id]:.3f})\n"
        )

    feature_idx = feature[node_id]
    feature_name = FEATURE_NAMES[feature_idx]
    standardized_threshold = threshold[node_id]
    raw_threshold = standardized_threshold_to_raw(
        feature_idx,
        standardized_threshold,
        stats,
    )

    left_rules = export_if_else_rules(
        structure,
        stats,
        children_left[node_id],
        indent + 1,
    )
    right_rules = export_if_else_rules(
        structure,
        stats,
        children_right[node_id],
        indent + 1,
    )

    return (
        f"{prefix}if {feature_name} <= {raw_threshold:.4f} "
        f"(standardized <= {standardized_threshold:.4f}):\n"
        f"{left_rules}"
        f"{prefix}else:  # {feature_name} > {raw_threshold:.4f}\n"
        f"{right_rules}"
    )


def train_and_extract(max_depth: int = 10, verbose: bool = True) -> TrainResult:
    """Pima dataset으로 깊은 Decision Tree를 학습하고 구조를 추출합니다."""
    x_raw, y = load_raw_dataset()

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_raw,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    stats = fit_preprocess_stats(x_train_raw)
    x_train = transform_features(x_train_raw, stats)
    x_test = transform_features(x_test_raw, stats)

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=2,
        random_state=42,
    )
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    if verbose:
        print(f"[일반 Decision Tree 정확도]: {acc * 100:.1f}%", flush=True)
        print(f"[테스트 샘플 수]: {len(x_test)}개", flush=True)
        print(f"[트리 max_depth 설정]: {max_depth}", flush=True)
        print(f"[학습된 트리 실제 depth]: {clf.get_depth()}", flush=True)
        print(f"[학습된 트리 leaf 수]: {clf.get_n_leaves()}", flush=True)

    structure = extract_tree_structure(clf)
    rules_text = export_if_else_rules(structure, stats)
    return TrainResult(
        clf=clf,
        structure=structure,
        x_test_raw=x_test_raw,
        x_test=x_test,
        y_test=y_test,
        stats=stats,
        rules_text=rules_text,
    )
