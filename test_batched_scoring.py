"""batched candidate scoring 정확성 + 속도 검증 (iris, 전부 한 batch에 들어가는 케이스)."""

from __future__ import annotations

import time

import numpy as np

from client_assisted.candidates import make_public_grid_candidates, make_small_public_threshold_grid
from client_assisted.dataset import load_scaled_dataset_subset, one_hot_encode
from fully_encrypted_mgi_stump import create_bootstrap_context, encrypt_dataset, evaluate_candidate
from batched_candidate_scoring import (
    evaluate_candidate_batch_counts,
    extract_block_value,
    max_batch_size,
    next_pow2,
)

X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(dataset_name="iris", test_size=30)
y_train_one_hot = one_hot_encode(y_train, n_classes=len(class_names))
candidates = make_public_grid_candidates(
    n_features=X_train.shape[1], thresholds=make_small_public_threshold_grid(candidate_count=3)
)
n_samples = X_train.shape[0]
print(f"[setup] candidates={len(candidates)} n_samples={n_samples}")

ctx = create_bootstrap_context(mode="gpu")
dataset = encrypt_dataset(ctx, X_train, y_train_one_hot)

block_size, max_k = max_batch_size(ctx, n_samples)
print(f"block_size={block_size} | 한 ciphertext에 들어가는 candidate 수={max_k}")

# --- 기존 방식: candidate마다 순차 sigmoid 호출 ---
t0 = time.time()
old_scores = []
for c in candidates:
    score, lc, rc = evaluate_candidate(ctx, dataset, c)
    old_scores.append(float(ctx.engine.decrypt(score, ctx.sk)[0].real))
old_time = time.time() - t0
print(f"\n[기존: candidate별 순차 sigmoid] time={old_time:.2f}s")
print(f"  scores={[f'{s:.2f}' for s in old_scores]}")

# --- 신규: 전부 한 batch로 packing해서 sigmoid 1번 ---
t0 = time.time()
left_counts, right_counts = evaluate_candidate_batch_counts(ctx, dataset, candidates, block_size)
new_time = time.time() - t0

new_scores = []
for i in range(len(candidates)):
    lc = np.array([extract_block_value(ctx, ct, i, block_size) for ct in left_counts])
    rc = np.array([extract_block_value(ctx, ct, i, block_size) for ct in right_counts])
    score = (lc.sum() ** 2 - (lc**2).sum()) + (rc.sum() ** 2 - (rc**2).sum())
    new_scores.append(score)
print(f"\n[신규: batched, sigmoid 1번] time={new_time:.2f}s")
print(f"  scores={[f'{s:.2f}' for s in new_scores]}")

print(f"\n최대 오차: {max(abs(a - b) for a, b in zip(old_scores, new_scores)):.4f}")
print(f"speedup: {old_time / new_time:.2f}x")
