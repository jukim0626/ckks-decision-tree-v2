from train_tree import train_and_extract
from ckks_tree import COMPARE_SCALE, create_ckks_context, predict_ckks
from sklearn.metrics import confusion_matrix
import numpy as np
import time


def print_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray | list[int],
    class_names: list[str],
    title: str = "Confusion Matrix",
) -> None:
    """정답/예측 라벨로 confusion matrix를 출력."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    # 긴 이름 대비 컬럼 폭을 동적으로 조정
    col_w = max(12, max(len(n) for n in class_names) + 2)
    print(f"\n[{title}]")
    print(f"{'':>{col_w}}", end="")
    for name in class_names:
        print(f"{name:>{col_w}}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>{col_w}}", end="")
        for val in row:
            print(f"{val:>{col_w}}", end="")
        print()


def print_compare_input_range(
    X_test: np.ndarray,
    structure: dict,
    compare_scale: float,
) -> None:
    """평문 기준으로 각 decision node의 sigmoid 비교 입력 범위를 출력."""
    children_left = structure["children_left"]
    feature = structure["feature"]
    threshold = structure["threshold"]
    leaf = -1

    global_min = None
    global_max = None

    print("\n[Compare input range 진단]")
    print(f"compare_scale: {compare_scale}")

    for node_id, left_child in enumerate(children_left):
        if left_child == leaf:
            continue

        feature_idx = feature[node_id]
        thresh = threshold[node_id]
        raw_diff = X_test[:, feature_idx] - thresh
        compare_input = compare_scale * raw_diff

        node_min = float(np.min(compare_input))
        node_max = float(np.max(compare_input))
        global_min = node_min if global_min is None else min(global_min, node_min)
        global_max = node_max if global_max is None else max(global_max, node_max)

        print(
            f"node {node_id} | feature={feature_idx} | threshold={thresh:.6f} | "
            f"raw_diff min/max=({np.min(raw_diff):.6f}, {np.max(raw_diff):.6f}) | "
            f"compare_input min/max=({node_min:.6f}, {node_max:.6f})"
        )

    if global_min is None or global_max is None:
        print("decision node가 없습니다.")
    else:
        print(
            "global compare_input min/max="
            f"({global_min:.6f}, {global_max:.6f})"
        )


def run_dataset(dataset_name: str, scaler_name: str, ctx: dict) -> None:
    """한 데이터셋에 대해 평문 DT vs CKKS DT 정확도 비교."""
    print(f"\n=========== dataset: {dataset_name} | scaler: {scaler_name} ===========")
    clf, structure, X_test, y_test, class_names = train_and_extract(
        dataset_name=dataset_name,
        scaler_name=scaler_name,
    )
    n_samples = len(X_test)
    n_features = X_test.shape[1]
    n_nodes = len(structure["children_left"])
    n_classes = len(class_names)

    print("\n[실험 정보]")
    print(f"Dataset 이름: {dataset_name}")
    print(f"Scaler 이름: {scaler_name}")
    print(f"클래스 이름: {class_names}")
    print(f"클래스 수: {n_classes}")
    print(f"테스트 샘플 수: {n_samples}")
    print(f"feature 수: {n_features}")
    print(f"tree node 수: {n_nodes}")

    # 일반 Decision Tree
    y_pred_plain = clf.predict(X_test)
    plain_acc = np.mean(y_pred_plain == y_test) * 100
    print_confusion_matrix(y_test, y_pred_plain, class_names, "일반 Decision Tree")

    print_compare_input_range(X_test, structure, COMPARE_SCALE)

    # CKKS 추론
    print("\nCKKS 클래스 분류 중...")
    y_pred_ckks = []
    total_start = time.time()
    for i in range(n_samples):
        sample = X_test[i]
        sample_start = time.time()
        print(f"[CKKS] sample {i + 1}/{n_samples} 시작", flush=True)
        pred = predict_ckks(ctx, sample, structure)
        y_pred_ckks.append(pred)
        sample_elapsed = time.time() - sample_start
        print(
            f"[CKKS] sample {i + 1}/{n_samples} 완료 | "
            f"pred={pred} | true={y_test[i]} | time={sample_elapsed:.2f}s",
            flush=True,
        )

    total_elapsed = time.time() - total_start
    ckks_acc = np.mean(np.array(y_pred_ckks) == y_test) * 100
    match_count = int(np.sum(y_pred_plain == np.array(y_pred_ckks)))

    print_confusion_matrix(y_test, y_pred_ckks, class_names, "CKKS Decision Tree")

    # 결과 비교
    print("\n--- 결과 비교 ---")
    print(f"Dataset 이름: {dataset_name}")
    print(f"Scaler 이름: {scaler_name}")
    print(f"클래스 이름: {class_names}")
    print(f"클래스 수: {n_classes}")
    print(f"테스트 샘플 수: {n_samples}")
    print(f"feature 수: {n_features}")
    print(f"tree node 수: {n_nodes}")
    print(f"Plain DT accuracy: {plain_acc:.2f}%")
    print(f"CKKS DT accuracy: {ckks_acc:.2f}%")
    print(f"Plain/CKKS match count: {match_count}/{n_samples}")
    print(f"CKKS total time: {total_elapsed:.2f}s")
    print(f"CKKS per sample time: {total_elapsed / n_samples:.2f}s")
    print(f"정확도 차이 : {abs(plain_acc - ckks_acc):.1f}%p")


def main():
    # CKKS context는 데이터셋과 무관하므로 한 번만 생성해서 재사용
    ctx = create_ckks_context()

    experiments = [
        ("wine", "minmax_minus1_1"),
    ]

    # experiments = [
    #     ("iris", "none"),
    #     ("wine", "none"),
    #     ("wine", "minmax_minus1_1"),
    #     ("breast_cancer", "none"),
    #     ("breast_cancer", "minmax_minus1_1"),
    # ]

    for dataset_name, scaler_name in experiments:
        run_dataset(dataset_name, scaler_name, ctx)


if __name__ == "__main__":
    main()
