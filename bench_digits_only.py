"""digits(10-class, 64 feature) fully-encrypted MGI SIMD 재실행 (GPU OOM 재시도)."""

from __future__ import annotations

import time

from client_assisted.candidates import make_public_grid_candidates, make_small_public_threshold_grid
from client_assisted.dataset import load_scaled_dataset_subset, one_hot_encode
from fully_encrypted_mgi_stump import create_bootstrap_context, encrypt_dataset, plaintext_mgi_best_split
from fully_encrypted_mgi_simd_argmin import debug_decrypt_simd_winner, train_fully_encrypted_mgi_stump_simd

X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(dataset_name="digits", test_size=30)
y_train_one_hot = one_hot_encode(y_train, n_classes=len(class_names))
candidates = make_public_grid_candidates(
    n_features=X_train.shape[1], thresholds=make_small_public_threshold_grid(candidate_count=3)
)
print(f"[setup] candidates={len(candidates)} | train_samples={X_train.shape[0]} | n_features={X_train.shape[1]}", flush=True)

best_idx, best_score = plaintext_mgi_best_split(X_train, y_train_one_hot, candidates)
print(f"[plaintext MGI] best={candidates[best_idx]} score={best_score:.4f}", flush=True)

ctx = create_bootstrap_context(mode="gpu")
dataset = encrypt_dataset(ctx, X_train, y_train_one_hot)

import sys
from sharpen_iterations_formula import required_sharpen_iterations

SHARPEN = int(sys.argv[1]) if len(sys.argv) > 1 else 12
print(f"[config] sharpen_iterations={SHARPEN} (공식 추천: k(0.95)={required_sharpen_iterations(0.001236, 0.95)})", flush=True)

t0 = time.time()
winner = train_fully_encrypted_mgi_stump_simd(ctx, dataset, candidates, sharpen_iterations=SHARPEN, verbose=True)
elapsed = time.time() - t0
decoded = debug_decrypt_simd_winner(ctx, winner)
picked_feature = max(range(len(decoded["feature_selectors"])), key=lambda i: decoded["feature_selectors"][i])
print(f"\n[fully-encrypted simd] time={elapsed:.2f}s | feature={picked_feature} "
      f"(confidence={decoded['feature_selectors'][picked_feature]:.3f}) | threshold={decoded['threshold']:.4f}")
print(f"plaintext best feature={candidates[best_idx].feature_idx}")
