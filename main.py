from train_tree import train_and_extract
from ckks_iris import create_ckks_context, predict_ckks
from sklearn.metrics import confusion_matrix
import numpy as np
import time


def print_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    class_names = ["setosa", "versicolor", "virginica"]
    print(f"\n[{title}]")
    print(f"{'':>12}", end="")
    for name in class_names:
        print(f"{name:>12}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>12}", end="")
        for val in row:
            print(f"{val:>12}", end="")
        print()


def main():
    clf, structure, X_test, y_test = train_and_extract()

    # 일반 Decision Tree
    y_pred_plain = clf.predict(X_test)
    plain_acc = np.mean(y_pred_plain == y_test) * 100
    print_confusion_matrix(y_test, y_pred_plain, "일반 Decision Tree")

    # CKKS context 생성
    ctx = create_ckks_context()

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

    print_confusion_matrix(y_test, y_pred_ckks, "CKKS Decision Tree")

    # 결과 비교
    print("\n--- 결과 비교 ---")
    print(f"일반 DT : {plain_acc:.1f}%")
    print(f"CKKS DT : {ckks_acc:.1f}%")
    print(f"정확도 차이 : {abs(plain_acc - ckks_acc):.1f}%p")
    print(f"총 소요시간 : {total_elapsed:.2f}s ({total_elapsed / n_samples:.2f}s/sample)")


if __name__ == "__main__":
    main()