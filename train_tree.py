from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def train_and_extract(dataset_name: str = "iris"):
    """지정한 sklearn 데이터셋으로 Decision Tree를 학습하고 트리 구조를 dict로 반환.

    Returns:
        clf         : 학습된 DecisionTreeClassifier
        structure   : {"children_left", "children_right", "feature", "threshold", "value"}
        X_test      : 테스트 입력
        y_test      : 테스트 라벨
        class_names : 데이터셋의 클래스 이름 리스트 (str)
    """
    if dataset_name not in _LOADERS:
        raise ValueError(
            f"unknown dataset: {dataset_name!r}. "
            f"choose from {list(_LOADERS.keys())}"
        )

    # 데이터 로드
    data = _LOADERS[dataset_name]()
    X, y = data.data, data.target
    class_names = [str(n) for n in data.target_names]

    # 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Decision Tree 학습
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # 정확도 확인
    acc = clf.score(X_test, y_test)
    print(f"[일반 Decision Tree 정확도 - {dataset_name}]: {acc * 100:.1f}%")
    print(f"[테스트 샘플 수]: {len(X_test)}개")

    # 트리 구조 추출
    tree = clf.tree_
    structure = {
        "children_left":  tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature":        tree.feature.tolist(),
        "threshold":      tree.threshold.tolist(),
        "value":          tree.value.tolist(),
    }
    # 구조 확인용 출력
    print("\n=== TREE 구조 ===")
    print("n_nodes:", tree.node_count)
    print("children_left :", tree.children_left.tolist())
    print("children_right:", tree.children_right.tolist())
    print("feature       :", tree.feature.tolist())
    print("threshold     :", [round(t, 2) for t in tree.threshold.tolist()])
    for i in range(tree.node_count):
        print(f"node {i}: value =", tree.value[i].tolist())

    return clf, structure, X_test, y_test, class_names
