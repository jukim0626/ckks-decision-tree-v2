"""Run plain and DESILO FHE Decision Tree inference on Pima Diabetes."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from pima_diabetes.fhe_diabetes import create_ckks_context, predict_ckks_binary
from pima_diabetes.train_tree import train_and_extract


CLASS_NAMES = ["normal", "diabetes"]


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    """confusion matrix를 terminal에 보기 좋게 출력합니다."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n[{title}]", flush=True)
    print(f"{'':>12}", end="")
    for name in CLASS_NAMES:
        print(f"{name:>12}", end="")
    print(flush=True)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>12}", end="")
        for val in row:
            print(f"{val:>12}", end="")
        print(flush=True)


def save_rules(rules_text: str, max_depth: int, actual_depth: int) -> Path:
    """추출한 if/else rule을 Markdown 파일로 저장합니다."""
    output_path = (
        Path(__file__).resolve().parent
        / f"tree_rules_max_depth_{max_depth}_actual_depth_{actual_depth}.md"
    )
    output_path.write_text(
        "# Pima Diabetes Decision Tree Rules\n\n"
        f"- max_depth setting: `{max_depth}`\n"
        f"- actual tree depth: `{actual_depth}`\n\n"
        "```python\n"
        f"{rules_text}"
        "```\n",
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    """command line argument를 읽습니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--fhe-samples", type=int, default=3)
    parser.add_argument("--fhe-max-level", type=int, default=15)
    parser.add_argument("--run-fhe", action="store_true")
    parser.add_argument("--skip-fhe", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Pima Diabetes plain DT와 FHE soft DT 결과를 비교합니다."""
    args = parse_args()

    result = train_and_extract(max_depth=args.max_depth)
    rules_path = save_rules(
        result.rules_text,
        max_depth=args.max_depth,
        actual_depth=result.clf.get_depth(),
    )
    print(f"[if/else rule 저장]: {rules_path}", flush=True)

    y_pred_plain = result.clf.predict(result.x_test)
    plain_acc = np.mean(y_pred_plain == result.y_test) * 100
    print_confusion_matrix(result.y_test, y_pred_plain, "일반 Decision Tree")

    if args.skip_fhe or not args.run_fhe:
        print("\n[DESILO FHE 추론]: 기본 rule 생성 모드라 건너뜀", flush=True)
        print("FHE 추론을 실행하려면 --run-fhe 옵션을 추가하세요.", flush=True)
        print("\n--- 결과 비교 ---", flush=True)
        print(f"일반 DT : {plain_acc:.1f}%", flush=True)
        return

    print(f"\n[DESILO FHE max_level]: {args.fhe_max_level}", flush=True)
    print("[DESILO FHE context 생성 중...]", flush=True)
    ctx = create_ckks_context(max_level=args.fhe_max_level)

    print("\nDESILO FHE 클래스 분류 중...", flush=True)
    y_pred_ckks = []
    n_samples = min(args.fhe_samples, len(result.x_test))
    total_start = time.time()

    for i in range(n_samples):
        sample = result.x_test[i]
        pred = predict_ckks_binary(ctx, sample, result.structure)
        y_pred_ckks.append(pred)
        print(f"  - FHE sample {i + 1}/{n_samples} 완료", flush=True)

    total_elapsed = time.time() - total_start
    y_pred_ckks_array = np.array(y_pred_ckks)
    y_test_subset = result.y_test[:n_samples]
    ckks_acc = np.mean(y_pred_ckks_array == y_test_subset) * 100

    print_confusion_matrix(y_test_subset, y_pred_ckks_array, "DESILO FHE Decision Tree")

    print("\n--- 결과 비교 ---", flush=True)
    print(f"일반 DT 전체 test : {plain_acc:.1f}%", flush=True)
    print(f"FHE DT subset     : {ckks_acc:.1f}% ({n_samples} samples)", flush=True)
    print(f"총 소요시간 : {total_elapsed:.2f}s ({total_elapsed / n_samples:.2f}s/sample)", flush=True)


if __name__ == "__main__":
    main()
