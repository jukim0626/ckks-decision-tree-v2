"""score 정밀도 수정(원본 counts에서 한 번만 재계산) 전/후 비교. iris depth=1 기준."""

from __future__ import annotations

import time

from client_assisted.candidates import make_public_grid_candidates, make_small_public_threshold_grid
from client_assisted.dataset import load_scaled_dataset_subset, one_hot_encode
from fully_encrypted_mgi_stump import create_bootstrap_context, encrypt_dataset, plaintext_mgi_best_split
from fully_encrypted_mgi_simd_argmin import train_fully_encrypted_mgi_stump_simd_precise_score

X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(dataset_name="iris", test_size=30)
y_train_one_hot = one_hot_encode(y_train, n_classes=len(class_names))
candidates = make_public_grid_candidates(
    n_features=X_train.shape[1], thresholds=make_small_public_threshold_grid(candidate_count=3)
)
best_idx, best_score = plaintext_mgi_best_split(X_train, y_train_one_hot, candidates)
print(f"[plaintext MGI] best={candidates[best_idx]} score={best_score:.4f}")

ctx = create_bootstrap_context(mode="gpu")
dataset = encrypt_dataset(ctx, X_train, y_train_one_hot)

t0 = time.time()
winner = train_fully_encrypted_mgi_stump_simd_precise_score(
    ctx, dataset, candidates, sharpen_iterations=12, verbose=False
)
elapsed = time.time() - t0

score_blended = float(ctx.engine.decrypt(winner["score_blended"], ctx.sk)[0].real)
score_precise = float(ctx.engine.decrypt(winner["score_precise"], ctx.sk)[0].real)
matched = candidates[winner["matched_candidate_idx"]]

print(f"\ntime={elapsed:.2f}s | matched_candidate={matched}")
print(f"score_blended (기존 방식, 매 라운드 blend+bootstrap 누적): {score_blended:.4f}")
print(f"score_precise (새 방식, 원본 counts에서 1회 재계산):        {score_precise:.4f}")
print(f"plaintext(soft-MGI) 참고값: {best_score:.4f}")
print(f"\n오차: blended={abs(score_blended-best_score):.2f} | precise={abs(score_precise-best_score):.2f}")
