"""batched scoring을 전체 fully-encrypted MGI 파이프라인(scoring+SIMD argmin)에 통합해서
기존 evaluate_all_candidates_packed(순차 sigmoid) 대비 end-to-end 속도/정확도 비교."""

from __future__ import annotations

import time

from client_assisted.candidates import make_public_grid_candidates, make_small_public_threshold_grid
from client_assisted.dataset import load_scaled_dataset_subset, one_hot_encode
from fully_encrypted_mgi_stump import create_bootstrap_context, encrypt_dataset, plaintext_mgi_best_split
from fully_encrypted_mgi_simd_argmin import evaluate_all_candidates_packed, simd_reduce_argmin
from batched_candidate_scoring import evaluate_all_candidates_packed_batched

DATASET = "iris"
SHARPEN = 12

X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(dataset_name=DATASET, test_size=30)
y_train_one_hot = one_hot_encode(y_train, n_classes=len(class_names))
candidates = make_public_grid_candidates(
    n_features=X_train.shape[1], thresholds=make_small_public_threshold_grid(candidate_count=3)
)
best_idx, best_score = plaintext_mgi_best_split(X_train, y_train_one_hot, candidates)
print(f"[plaintext MGI] best={candidates[best_idx]} score={best_score:.4f}")
normalizer = float(X_train.shape[0] ** 2)

# --- 기존 파이프라인 ---
ctx1 = create_bootstrap_context(mode="gpu")
dataset1 = encrypt_dataset(ctx1, X_train, y_train_one_hot)
t0 = time.time()
packed_score1, aux1, n_pow2, n_features, n_classes = evaluate_all_candidates_packed(
    ctx1, dataset1, candidates, normalizer, verbose=False
)
winner_score1, winner_aux1 = simd_reduce_argmin(ctx1, packed_score1, aux1, n_pow2, normalizer, SHARPEN, verbose=False)
time1 = time.time() - t0
feat_conf1 = [float(ctx1.engine.decrypt(fc, ctx1.sk)[0].real) for fc in winner_aux1[1 : 1 + n_features]]
picked1 = max(range(n_features), key=lambda i: feat_conf1[i])
print(f"\n[기존 파이프라인] time={time1:.2f}s | picked_feature={picked1} (conf={feat_conf1[picked1]:.3f})")

# --- batched scoring 파이프라인 ---
ctx2 = create_bootstrap_context(mode="gpu")
dataset2 = encrypt_dataset(ctx2, X_train, y_train_one_hot)
t0 = time.time()
packed_score2, aux2, n_pow2b, n_features2, n_classes2 = evaluate_all_candidates_packed_batched(
    ctx2, dataset2, candidates, normalizer, verbose=True
)
winner_score2, winner_aux2 = simd_reduce_argmin(ctx2, packed_score2, aux2, n_pow2b, normalizer, SHARPEN, verbose=False)
time2 = time.time() - t0
feat_conf2 = [float(ctx2.engine.decrypt(fc, ctx2.sk)[0].real) for fc in winner_aux2[1 : 1 + n_features2]]
picked2 = max(range(n_features2), key=lambda i: feat_conf2[i])
print(f"\n[batched scoring 파이프라인] time={time2:.2f}s | picked_feature={picked2} (conf={feat_conf2[picked2]:.3f})")

print(f"\n=== 결과 ===")
print(f"plaintext best feature={candidates[best_idx].feature_idx}")
print(f"기존: {time1:.2f}s | batched: {time2:.2f}s | speedup={time1/time2:.2f}x")
