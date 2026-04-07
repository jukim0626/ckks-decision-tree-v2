from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def train_and_extract():
    # 데이터 로드
    iris = load_iris()
    X, y = iris.data, iris.target

    # 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Decision Tree 학습
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # 정확도 확인
    acc = clf.score(X_test, y_test)
    print(f"[일반 Decision Tree 정확도]: {acc * 100:.1f}%")
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
    return clf, structure, X_test, y_test