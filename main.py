from train_tree import train_and_extract
from ckks_tree import create_ckks_context, predict_ckks
from sklearn.metrics import confusion_matrix
import numpy as np
import time


def print_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
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


def run_dataset(dataset_name: str, ctx: dict) -> None:
    """한 데이터셋에 대해 평문 DT vs CKKS DT 정확도 비교."""
    print(f"\n=========== dataset: {dataset_name} ===========")
    clf, structure, X_test, y_test, class_names = train_and_extract(dataset_name)

    # 일반 Decision Tree
    y_pred_plain = clf.predict(X_test)
    plain_acc = np.mean(y_pred_plain == y_test) * 100
    print_confusion_matrix(y_test, y_pred_plain, class_names, "일반 Decision Tree")

    # CKKS 추론
    print("\nCKKS 클래스 분류 중...")
    y_pred_ckks = []
    n_samples = len(X_test)
    total_start = time.time()
    for i in range(n_samples):
        sample = X_test[i]
        pred = predict_ckks(ctx, sample, structure)
        y_pred_ckks.append(pred)

    total_elapsed = time.time() - total_start
    ckks_acc = np.mean(np.array(y_pred_ckks) == y_test) * 100

    print_confusion_matrix(y_test, y_pred_ckks, class_names, "CKKS Decision Tree")

    # 결과 비교
    print("\n--- 결과 비교 ---")
    print(f"일반 DT : {plain_acc:.1f}%")
    print(f"CKKS DT : {ckks_acc:.1f}%")
    print(f"정확도 차이 : {abs(plain_acc - ckks_acc):.1f}%p")
    print(f"총 소요시간 : {total_elapsed:.2f}s ({total_elapsed / n_samples:.2f}s/sample)")


def main():
    # CKKS context는 데이터셋과 무관하므로 한 번만 생성해서 재사용
    ctx = create_ckks_context()

    for dataset_name in ["iris", "wine", "breast_cancer"]:
        run_dataset(dataset_name, ctx)


if __name__ == "__main__":
    main()
