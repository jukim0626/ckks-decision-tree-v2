from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sigmoid_approx_coeffs import SIGMOID_COEFFS
import numpy as np


_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}

_SCALER_NAMES = {
    "none",
    "minmax_0_1",
    "minmax_minus1_1",
    "standard",
}

_THRESHOLD_TUNING_NAMES = {
    "none",
    "soft_surrogate",
}


def _create_scaler(scaler_name: str) -> MinMaxScaler | StandardScaler | None:
    """scaler 이름에 맞는 sklearn scaler 객체를 새로 생성."""
    if scaler_name == "none":
        return None
    if scaler_name == "minmax_0_1":
        return MinMaxScaler(feature_range=(0, 1))
    if scaler_name == "minmax_minus1_1":
        return MinMaxScaler(feature_range=(-1, 1))
    if scaler_name == "standard":
        return StandardScaler()
    raise ValueError(
        f"unknown scaler: {scaler_name!r}. "
        f"choose from {sorted(_SCALER_NAMES)}"
    )


def _sigmoid_approx_plain(x: np.ndarray) -> np.ndarray:
    """ckks_tree.py의 sigmoid polynomial 근사를 plaintext numpy로 계산."""
    return np.polynomial.polynomial.polyval(x, SIGMOID_COEFFS)


def _soft_predict_scores(X: np.ndarray, structure: dict) -> np.ndarray:
    """CKKS tree traversal과 같은 soft split 방식으로 class score 계산."""
    children_left = structure["children_left"]
    children_right = structure["children_right"]
    feature = structure["feature"]
    threshold = structure["threshold"]
    value = structure["value"]

    n_samples = X.shape[0]
    n_nodes = len(children_left)
    n_classes = len(value[0][0])
    leaf = -1

    node_weights = np.zeros((n_samples, n_nodes), dtype=float)
    node_weights[:, 0] = 1.0
    class_scores = np.zeros((n_samples, n_classes), dtype=float)

    for node_id in range(n_nodes):
        weights = node_weights[:, node_id]
        if children_left[node_id] == leaf:
            node_value = value[node_id][0]
            pred_class = int(np.argmax(node_value))
            class_scores[:, pred_class] += weights
            continue

        feature_idx = feature[node_id]
        raw_diff = X[:, feature_idx] - threshold[node_id]
        right_prob = _sigmoid_approx_plain(raw_diff)
        left_prob = 1.0 - right_prob

        node_weights[:, children_left[node_id]] += weights * left_prob
        node_weights[:, children_right[node_id]] += weights * right_prob

    return class_scores


def _hard_predict_from_structure(X: np.ndarray, structure: dict) -> np.ndarray:
    """structure threshold 기준의 hard Decision Tree 예측."""
    children_left = structure["children_left"]
    children_right = structure["children_right"]
    feature = structure["feature"]
    threshold = structure["threshold"]
    value = structure["value"]
    leaf = -1

    y_pred = []
    for sample in X:
        node_id = 0
        while children_left[node_id] != leaf:
            feature_idx = feature[node_id]
            if sample[feature_idx] <= threshold[node_id]:
                node_id = children_left[node_id]
            else:
                node_id = children_right[node_id]
        y_pred.append(int(np.argmax(value[node_id][0])))
    return np.array(y_pred)


def _threshold_objective(
    X: np.ndarray,
    y: np.ndarray,
    structure: dict,
) -> tuple[float, float, float, float]:
    """soft traversal accuracy를 중심으로 threshold 후보 점수 계산."""
    scores = _soft_predict_scores(X, structure)
    soft_pred = np.argmax(scores, axis=1)
    soft_acc = float(np.mean(soft_pred == y))

    true_scores = scores[np.arange(len(y)), y]
    other_scores = scores.copy()
    other_scores[np.arange(len(y)), y] = -np.inf
    soft_margin = float(np.mean(true_scores - np.max(other_scores, axis=1)))

    hard_pred = _hard_predict_from_structure(X, structure)
    hard_acc = float(np.mean(hard_pred == y))

    # soft_acc가 1순위, margin은 동률 해소, hard_acc는 과도한 붕괴 방지용.
    objective = soft_acc + 0.02 * soft_margin + 0.05 * hard_acc
    return objective, soft_acc, soft_margin, hard_acc


def _candidate_thresholds(
    feature_values: np.ndarray,
    original_threshold: float,
    max_candidates: int = 200,
) -> np.ndarray:
    """training feature 값 사이의 midpoint로 threshold 후보 생성."""
    unique_values = np.unique(feature_values)
    if len(unique_values) < 2:
        return np.array([original_threshold], dtype=float)

    midpoints = (unique_values[:-1] + unique_values[1:]) / 2.0
    if len(midpoints) > max_candidates:
        indices = np.linspace(0, len(midpoints) - 1, max_candidates, dtype=int)
        midpoints = midpoints[indices]
    return np.unique(np.append(midpoints, original_threshold))


def _tune_thresholds_for_soft_surrogate(
    structure: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_passes: int = 4,
) -> list[dict]:
    """학습 데이터에서 soft traversal 정확도가 좋아지도록 threshold 보정."""
    children_left = structure["children_left"]
    feature = structure["feature"]
    leaf = -1
    history = []

    structure["original_threshold"] = list(structure["threshold"])
    before = _threshold_objective(X_train, y_train, structure)

    for pass_idx in range(max_passes):
        improved = False
        for node_id, left_child in enumerate(children_left):
            if left_child == leaf:
                continue

            feature_idx = feature[node_id]
            old_threshold = structure["threshold"][node_id]
            candidates = _candidate_thresholds(
                X_train[:, feature_idx],
                old_threshold,
            )

            best_threshold = old_threshold
            best_score = _threshold_objective(X_train, y_train, structure)

            for candidate in candidates:
                structure["threshold"][node_id] = float(candidate)
                candidate_score = _threshold_objective(X_train, y_train, structure)
                if candidate_score > best_score:
                    best_threshold = float(candidate)
                    best_score = candidate_score

            structure["threshold"][node_id] = best_threshold
            if abs(best_threshold - old_threshold) > 1e-12:
                improved = True
                history.append(
                    {
                        "pass": pass_idx,
                        "node_id": node_id,
                        "feature": feature_idx,
                        "old_threshold": old_threshold,
                        "new_threshold": best_threshold,
                        "soft_acc": best_score[1],
                        "soft_margin": best_score[2],
                        "hard_acc": best_score[3],
                    }
                )

        if not improved:
            break

    after = _threshold_objective(X_train, y_train, structure)
    structure["threshold_tuning_summary"] = {
        "before_soft_acc": before[1],
        "before_soft_margin": before[2],
        "before_hard_acc": before[3],
        "after_soft_acc": after[1],
        "after_soft_margin": after[2],
        "after_hard_acc": after[3],
    }
    structure["threshold_tuning_history"] = history
    return history


def train_and_extract(
    dataset_name: str = "iris",
    scaler_name: str = "none",
    threshold_tuning: str = "none",
) -> tuple[DecisionTreeClassifier, dict, np.ndarray, np.ndarray, list[str]]:
    """지정한 sklearn 데이터셋으로 Decision Tree를 학습하고 트리 구조를 dict로 반환.

    Returns:
        clf         : 학습된 DecisionTreeClassifier
        structure   : {"children_left", "children_right", "feature", "threshold", "value"}
        X_test      : 테스트 입력 (scaler_name이 none이 아니면 scaling 적용됨)
        y_test      : 테스트 라벨
        class_names : 데이터셋의 클래스 이름 리스트 (str)
    """
    if dataset_name not in _LOADERS:
        raise ValueError(
            f"unknown dataset: {dataset_name!r}. "
            f"choose from {list(_LOADERS.keys())}"
        )
    if scaler_name not in _SCALER_NAMES:
        raise ValueError(
            f"unknown scaler: {scaler_name!r}. "
            f"choose from {sorted(_SCALER_NAMES)}"
        )
    if threshold_tuning not in _THRESHOLD_TUNING_NAMES:
        raise ValueError(
            f"unknown threshold_tuning: {threshold_tuning!r}. "
            f"choose from {sorted(_THRESHOLD_TUNING_NAMES)}"
        )

    # 데이터 로드
    data = _LOADERS[dataset_name]()
    X, y = data.data, data.target
    class_names = [str(n) for n in data.target_names]

    # 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling은 train에만 fit하고 test에는 transform만 적용
    scaler = _create_scaler(scaler_name)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Decision Tree 학습
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    plain_acc = clf.score(X_test, y_test)

    # 트리 구조 추출
    tree = clf.tree_
    structure = {
        "children_left":  tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature":        tree.feature.tolist(),
        "threshold":      tree.threshold.tolist(),
        "value":          tree.value.tolist(),
    }

    if threshold_tuning == "soft_surrogate":
        _tune_thresholds_for_soft_surrogate(
            structure=structure,
            X_train=X_train,
            y_train=y_train,
        )

    # 정확도 확인
    structure["threshold_tuning"] = threshold_tuning
    structure["plain_test_accuracy"] = plain_acc
    print(
        f"[원본 Plain Decision Tree 정확도 - {dataset_name}, "
        f"scaler={scaler_name}, threshold_tuning={threshold_tuning}]: "
        f"{plain_acc * 100:.1f}%"
    )
    print(f"[테스트 샘플 수]: {len(X_test)}개")

    return clf, structure, X_test, y_test, class_names
