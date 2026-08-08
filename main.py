"""EncryptedFixedDepthTreeModel(임의 depth)용 fully encrypted inference.

새 sample을 encrypt -> 한 번도 decrypt하지 않고 root부터 leaf까지 soft traversal ->
최종 class_scores만 client가 decrypt해서 plaintext argmax (논문 Section 4.1 방식).

encrypt_inference_sample()/encrypted_predict_class()/debug_decrypt_class_scores()는
depth와 무관한 "sample 하나 encrypt", "score decrypt+argmax" 단계라
client_assisted.inference에 있는 걸 그대로 재사용한다 (중복 구현 안 함).

--depth 2로 실행하면 예전 encrypted_depth2_inference.py(삭제됨)와 동일한 결과를 낸다
(EXPERIMENT_LOG.md 2026-07-07 참고, 100% 일치 재확인 후 depth2 전용 코드를 정리했다).
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np
from tqdm import tqdm

from client_assisted import (
    EncryptedDataset,
    EncryptedFixedDepthTreeModel,
    EncryptedTrainingContext,
    create_context,
    debug_decrypt_class_scores,
    debug_decrypt_fixed_depth_leaf_counts,
    encrypt_dataset,
    encrypt_inference_sample,
    encrypted_predict_class,
    format_float_list,
    load_scaled_dataset_subset,
    make_public_grid_candidates,
    make_small_public_threshold_grid,
    one_hot_encode,
    predict_fixed_depth_soft_plaintext,
    server_compute_weighted_child_weights,
    sync_engine_if_needed,
    train_client_assisted_fixed_depth_tree,
)


def encrypted_traverse_and_predict_fixed_depth(
    ctx: EncryptedTrainingContext,
    model: EncryptedFixedDepthTreeModel,
    enc_sample: list,
) -> list:
    """root부터 model.depth까지 encrypted 상태로 재귀적으로 soft traversal.

    client_assisted.verification.plaintext_fixed_depth_leaf_weights()의 재귀 walk와
    같은 구조이지만, sigmoid를 plaintext로 계산하는 대신 model.node_splits(encrypted
    split)와 server_compute_weighted_child_weights로 encrypted 상태를 유지한다.
    이 함수 내부에서는 decrypt를 절대 호출하지 않는다.
    """
    sample_dataset = EncryptedDataset(
        enc_features=enc_sample,
        enc_labels=[],
        n_samples=1,
        n_features=len(enc_sample),
        n_classes=0,
    )

    leaf_weights: list = []
    split_idx = 0

    def walk(enc_node_weight, current_depth: int) -> None:
        nonlocal split_idx
        if current_depth == model.depth:
            leaf_weights.append(enc_node_weight)
            return

        split = model.node_splits[split_idx]
        split_idx += 1
        left_weight, right_weight = server_compute_weighted_child_weights(
            ctx,
            sample_dataset,
            split,
            enc_node_weight,
        )
        del enc_node_weight
        walk(left_weight, current_depth + 1)
        walk(right_weight, current_depth + 1)

    root_weight = ctx.engine.encrypt([1.0], ctx.pk)
    walk(root_weight, current_depth=0)

    n_classes = len(model.leaf_counts[0])
    class_scores: list = [None] * n_classes
    for leaf_weight, leaf_counts in zip(leaf_weights, model.leaf_counts):
        for class_idx, leaf_count in enumerate(leaf_counts):
            piece = ctx.engine.multiply(leaf_weight, leaf_count, ctx.rlk)
            if class_scores[class_idx] is None:
                class_scores[class_idx] = piece
            else:
                class_scores[class_idx] = ctx.engine.add(class_scores[class_idx], piece)

    del leaf_weights
    return class_scores


def default_max_level_for_depth(depth: int) -> int:
    """--max-level을 안 주면 depth에 맞는 안전한 기본값을 고른다.

    depth>=3은 iris/wine/breast_cancer × depth=3,4,5 전체 9개 조합(candidate_count
    기본값 3 그대로)을 max_level=15로 실측 검증함 (EXPERIMENT_LOG.md 2026-07-07,
    "candidate 스트리밍 리팩터링" 이후 재테스트 참고). candidate를 server_ops.py+
    client_ops.py+training.py에서 "전부 계산 후 선택"이 아니라 "하나씩 계산->점수 확인
    ->즉시 폐기" 스트리밍 방식으로 바꾼 뒤로 peak 메모리가 O(candidates)에서 O(1)로
    줄어서, 예전에 데이터셋/depth마다 다르게 튜닝해야 했던 max_level이 15 하나로
    통일됐다 (breast_cancer depth=5, candidate=90개 그대로도 OOM 없이 성공).
    depth<=2는 이 재테스트 범위 밖이라 예전 값을 그대로 유지.
    """
    if depth <= 1:
        return 20
    if depth == 2:
        return 25
    return 15


def run_fixed_depth_encrypted_inference(
    dataset_name: str = "iris",
    depth: int = 3,
    test_size: int = 30,
    candidate_count: int = 3,
    max_level: int | None = None,
    mode: str = "gpu",
    slot_count: int | None = None,
) -> None:
    """test_size(기본 30개, 고정)만 test로 떼어내고 dataset 나머지 전체를 train으로 써서
    fixed-depth 모델을 학습하고, test set 전체를 encrypted inference로 검증.

    max_level=None이면 default_max_level_for_depth(depth)로 depth에 맞는 안전한 값을 쓴다.
    """
    if max_level is None:
        max_level = default_max_level_for_depth(depth)

    X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(
        dataset_name=dataset_name,
        test_size=test_size,
    )
    y_train_one_hot = one_hot_encode(y_train, n_classes=len(class_names))
    candidates = make_public_grid_candidates(
        n_features=X_train.shape[1],
        thresholds=make_small_public_threshold_grid(candidate_count=candidate_count),
    )

    print(
        f"[setup] dataset={dataset_name} | depth={depth} | candidates={len(candidates)} | "
        f"train_samples={X_train.shape[0]} | test_samples={X_test.shape[0]} | "
        f"n_features={X_train.shape[1]} | max_level={max_level} | mode={mode} | "
        f"slot_count={slot_count if slot_count is not None else 'auto'}",
        flush=True,
    )

    ctx = create_context(mode=mode, max_level=max_level, slot_count=slot_count)
    dataset = encrypt_dataset(ctx, X_train, y_train_one_hot)

    train_start = time.time()
    model, selections = train_client_assisted_fixed_depth_tree(
        ctx, dataset, candidates, depth=depth
    )
    sync_engine_if_needed(ctx)
    train_time = time.time() - train_start
    print(
        f"[training] mode={mode} | total={train_time:.2f}s ({train_time / 60:.2f}min) | "
        f"nodes={len(model.node_splits)} | {train_time / len(model.node_splits):.3f}s/node",
        flush=True,
    )

    encrypted_leaf_counts = debug_decrypt_fixed_depth_leaf_counts(ctx, model)
    plaintext_pred = predict_fixed_depth_soft_plaintext(
        X_test,
        selections,
        encrypted_leaf_counts,
    )

    print(f"\n[{dataset_name} depth={depth} encrypted inference]", flush=True)
    print(
        f"mode={ctx.mode} | candidates={len(candidates)} | test_samples={X_test.shape[0]} | n_features={X_train.shape[1]}",
        flush=True,
    )
    print(f"split_nodes={len(model.node_splits)} | leaf_nodes={len(model.leaf_counts)}", flush=True)

    encrypted_pred = []
    sample_times = []
    n_test = X_test.shape[0]
    for i in tqdm(range(n_test), desc="[inference] samples"):
        start = time.time()
        enc_sample = encrypt_inference_sample(ctx, X_test[i], ctx.pk)
        class_scores = encrypted_traverse_and_predict_fixed_depth(ctx, model, enc_sample)
        pred_class = encrypted_predict_class(ctx, class_scores, ctx.sk)
        sync_engine_if_needed(ctx)
        del enc_sample, class_scores
        gc.collect()
        elapsed = time.time() - start
        sample_times.append(elapsed)
        encrypted_pred.append(pred_class)
        tqdm.write(
            f"[inference] sample {i + 1}/{n_test} | "
            f"pred={pred_class} | true={y_test[i]} | time={elapsed:.3f}s"
        )

    encrypted_pred_arr = np.array(encrypted_pred)
    match_rate = float(np.mean(encrypted_pred_arr == plaintext_pred))
    encrypted_test_acc = float(np.mean(encrypted_pred_arr == y_test))
    plaintext_test_acc = float(np.mean(plaintext_pred == y_test))
    total_time = float(np.sum(sample_times))

    print("\n--- 결과 비교 ---", flush=True)
    print(f"class_names={class_names}")
    print(f"encrypted vs plaintext 예측 일치율: {match_rate * 100:.2f}%")
    print(f"encrypted test accuracy: {encrypted_test_acc * 100:.2f}%")
    print(f"plaintext test accuracy (predict_fixed_depth_soft_plaintext): {plaintext_test_acc * 100:.2f}%")
    print(f"encrypted inference 총 소요시간: {total_time:.2f}s ({total_time / len(sample_times):.3f}s/sample)")

    mismatch_indices = np.where(encrypted_pred_arr != plaintext_pred)[0]
    if len(mismatch_indices) > 0:
        print(f"\n불일치 sample 수: {len(mismatch_indices)} | indices={mismatch_indices.tolist()}")
        for idx in mismatch_indices:
            enc_sample = encrypt_inference_sample(ctx, X_test[idx], ctx.pk)
            class_scores = encrypted_traverse_and_predict_fixed_depth(ctx, model, enc_sample)
            decrypted_scores = debug_decrypt_class_scores(ctx, class_scores)
            print(
                f"  idx={idx} | encrypted class_scores decrypt={format_float_list(np.array(decrypted_scores))} | "
                f"plaintext_pred={int(plaintext_pred[idx])} | encrypted_pred={int(encrypted_pred_arr[idx])}"
            )
    else:
        print("\nencrypted 예측과 plaintext 예측이 test set 전체에서 100% 일치.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="fixed-depth(임의 depth) client-assisted encrypted training + encrypted inference"
    )
    parser.add_argument("--dataset", default="iris", choices=["iris", "wine", "breast_cancer"])
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--test-size",
        type=int,
        default=30,
        help="test set 개수 (고정). train은 dataset 나머지 전체를 사용",
    )
    parser.add_argument("--candidate-count", type=int, default=3, help="feature당 threshold candidate 개수")
    parser.add_argument(
        "--max-level",
        type=int,
        default=None,
        help="안 주면 depth에 맞는 안전한 기본값 사용 (depth=2->25, depth=3->30)",
    )
    parser.add_argument(
        "--mode",
        default="gpu",
        choices=["gpu", "cpu"],
        help="desilofhe Engine backend (논문의 single-core CPU 조건과 비교하려면 cpu)",
    )
    parser.add_argument(
        "--slot-count",
        type=int,
        default=None,
        help="ciphertext 슬롯 수 고정 (논문은 SEAL poly modulus degree 8192 -> 4096 packing 사용)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_fixed_depth_encrypted_inference(
        dataset_name=args.dataset,
        depth=args.depth,
        test_size=args.test_size,
        candidate_count=args.candidate_count,
        max_level=args.max_level,
        mode=args.mode,
        slot_count=args.slot_count,
    )
