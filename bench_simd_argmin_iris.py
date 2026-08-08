"""iris depth=1 stump에서 O(n) 순차 tournament argmin vs O(log n) SIMD 기반 argmin의
시간 차이를 직접 비교."""

from __future__ import annotations

import time

from client_assisted.candidates import make_public_grid_candidates, make_small_public_threshold_grid
from client_assisted.dataset import load_scaled_dataset_subset, one_hot_encode
from fully_encrypted_mgi_stump import create_bootstrap_context, encrypt_dataset, plaintext_mgi_best_split
from fully_encrypted_mgi_simd_argmin import debug_decrypt_simd_winner, train_fully_encrypted_mgi_stump_simd

import sys

CANDIDATE_COUNT = 3
SHARPEN_ITERATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def main() -> None:
    X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(dataset_name="iris", test_size=30)
    y_train_one_hot = one_hot_encode(y_train, n_classes=len(class_names))
    candidates = make_public_grid_candidates(
        n_features=X_train.shape[1],
        thresholds=make_small_public_threshold_grid(candidate_count=CANDIDATE_COUNT),
    )
    print(f"[setup] candidates={len(candidates)} | train_samples={X_train.shape[0]}", flush=True)

    best_idx, best_score = plaintext_mgi_best_split(X_train, y_train_one_hot, candidates)
    print(f"[plaintext MGI] best={candidates[best_idx]} score={best_score:.4f}", flush=True)

    ctx = create_bootstrap_context(mode="gpu")
    dataset = encrypt_dataset(ctx, X_train, y_train_one_hot)

    t0 = time.time()
    winner = train_fully_encrypted_mgi_stump_simd(
        ctx, dataset, candidates, sharpen_iterations=SHARPEN_ITERATIONS, verbose=True
    )
    elapsed = time.time() - t0
    decoded = debug_decrypt_simd_winner(ctx, winner)

    picked_feature = max(range(len(decoded["feature_selectors"])), key=lambda i: decoded["feature_selectors"][i])
    print(f"\n[simd fully-encrypted] time={elapsed:.2f}s", flush=True)
    print(f"  feature_selectors={[f'{v:.3f}' for v in decoded['feature_selectors']]}")
    print(f"  picked_feature={picked_feature} (plaintext best feature={candidates[best_idx].feature_idx})")
    print(f"  threshold(decrypt)={decoded['threshold']:.4f}")
    print(f"  score(decrypt)={decoded['score']:.4f}")
    print(f"\n비교: O(n) tournament(sharpen=20)=727.82s | SIMD(log2, sharpen=20)={elapsed:.2f}s | speedup={727.82/elapsed:.2f}x")


if __name__ == "__main__":
    main()
