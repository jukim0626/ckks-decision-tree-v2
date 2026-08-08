"""iris/wine/breast_cancer/diabetes 전부에서 depth=1 stump 기준
client-assisted(decrypt 기반) vs fully-encrypted MGI(SIMD argmin, sharpen=12,
merge_bootstrap 적용)의 시간/정확도를 비교해서 표로 정리.
"""

from __future__ import annotations

import time

from client_assisted import (
    create_context,
    encrypt_dataset as ca_encrypt_dataset,
    load_scaled_dataset_subset,
    make_public_grid_candidates,
    make_small_public_threshold_grid,
    one_hot_encode,
    train_client_assisted_fixed_depth_tree,
)
from fully_encrypted_mgi_stump import create_bootstrap_context, encrypt_dataset as mgi_encrypt_dataset, plaintext_mgi_best_split
from fully_encrypted_mgi_simd_argmin import debug_decrypt_simd_winner, train_fully_encrypted_mgi_stump_simd

CANDIDATE_COUNT = 3
SHARPEN_ITERATIONS = 12
DATASETS = ["iris", "wine", "breast_cancer", "diabetes", "digits"]

results = []

for dataset_name in DATASETS:
    print(f"\n{'=' * 60}\n{dataset_name}\n{'=' * 60}", flush=True)
    X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(
        dataset_name=dataset_name, test_size=30
    )
    y_train_one_hot = one_hot_encode(y_train, n_classes=len(class_names))
    candidates = make_public_grid_candidates(
        n_features=X_train.shape[1],
        thresholds=make_small_public_threshold_grid(candidate_count=CANDIDATE_COUNT),
    )
    n_candidates = len(candidates)
    print(
        f"[setup] candidates={n_candidates} | train_samples={X_train.shape[0]} | n_features={X_train.shape[1]}",
        flush=True,
    )

    best_idx, best_plain_score = plaintext_mgi_best_split(X_train, y_train_one_hot, candidates)
    best_candidate = candidates[best_idx]
    tied_indices = []
    for idx, c in enumerate(candidates):
        x = X_train[:, c.feature_idx]
        right = (x > c.threshold).astype(float)
        left = 1.0 - right
        lc = left @ y_train_one_hot
        rc = right @ y_train_one_hot
        s = (lc.sum() ** 2 - (lc**2).sum()) + (rc.sum() ** 2 - (rc**2).sum())
        if abs(s - best_plain_score) < 1e-6:
            tied_indices.append(idx)
    tied_features = sorted({candidates[i].feature_idx for i in tied_indices})
    print(
        f"[plaintext MGI] best={best_candidate} score={best_plain_score:.4f} | "
        f"tied_candidates={len(tied_indices)} (features={tied_features})",
        flush=True,
    )

    # client-assisted baseline
    ctx_ca = create_context(mode="gpu", max_level=20)
    dataset_ca = ca_encrypt_dataset(ctx_ca, X_train, y_train_one_hot)
    t0 = time.time()
    model, selections = train_client_assisted_fixed_depth_tree(
        ctx_ca, dataset_ca, candidates, depth=1, verbose=False
    )
    ca_time = time.time() - t0
    sel = selections.selections[0]
    ca_feature_ok = sel.feature_idx in tied_features
    print(
        f"[client-assisted] time={ca_time:.2f}s | feature={sel.feature_idx} | threshold={sel.threshold:.4f} | "
        f"correct_feature={ca_feature_ok}",
        flush=True,
    )
    del ctx_ca, dataset_ca

    # fully-encrypted MGI (SIMD argmin)
    ctx_mgi = create_bootstrap_context(mode="gpu")
    dataset_mgi = mgi_encrypt_dataset(ctx_mgi, X_train, y_train_one_hot)
    t0 = time.time()
    winner = train_fully_encrypted_mgi_stump_simd(
        ctx_mgi, dataset_mgi, candidates, sharpen_iterations=SHARPEN_ITERATIONS, verbose=False
    )
    mgi_time = time.time() - t0
    decoded = debug_decrypt_simd_winner(ctx_mgi, winner)
    picked_feature = max(
        range(len(decoded["feature_selectors"])), key=lambda i: decoded["feature_selectors"][i]
    )
    picked_confidence = decoded["feature_selectors"][picked_feature]
    mgi_feature_ok = picked_feature in tied_features
    print(
        f"[fully-encrypted simd] time={mgi_time:.2f}s | feature={picked_feature} "
        f"(confidence={picked_confidence:.3f}) | threshold={decoded['threshold']:.4f} | "
        f"correct_feature={mgi_feature_ok}",
        flush=True,
    )
    del ctx_mgi, dataset_mgi

    results.append(
        {
            "dataset": dataset_name,
            "n_candidates": n_candidates,
            "n_features": X_train.shape[1],
            "ca_time": ca_time,
            "ca_feature": sel.feature_idx,
            "ca_correct": ca_feature_ok,
            "mgi_time": mgi_time,
            "mgi_feature": picked_feature,
            "mgi_confidence": picked_confidence,
            "mgi_correct": mgi_feature_ok,
            "slowdown": mgi_time / ca_time,
        }
    )

print("\n\n" + "=" * 90)
print("요약 표")
print("=" * 90)
header = f"{'dataset':14s} | {'candidates':10s} | {'client-assisted':16s} | {'fully-encrypted':22s} | {'배율':8s}"
print(header)
print("-" * len(header))
for r in results:
    ca_str = f"{r['ca_time']:.1f}s (f={r['ca_feature']})"
    mgi_str = f"{r['mgi_time']:.1f}s (f={r['mgi_feature']},conf={r['mgi_confidence']:.2f})"
    print(f"{r['dataset']:14s} | {r['n_candidates']:10d} | {ca_str:16s} | {mgi_str:22s} | {r['slowdown']:.1f}x")

print("\ncorrect_feature(plaintext MGI 동점군과 일치 여부):")
for r in results:
    print(f"  {r['dataset']:14s} | client-assisted={r['ca_correct']} | fully-encrypted={r['mgi_correct']}")
