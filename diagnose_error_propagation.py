"""root 노드의 실제(불완전한) encrypted split이 자식 노드가 마주하는 candidate score/gap을
얼마나 왜곡시키는지 진단.

root 노드만 encrypted로 계산해서 실제 threshold + feature confidence 4개 전부를 얻은 뒤,
plaintext에서 두 가지 시나리오로 child(depth=1, left) weight를 계산해서 비교한다:
(a) "실제" - server_compute_weighted_child_weights와 동일하게, feature confidence로
    가중합한 selected_feature 사용 (soft, 여러 feature가 섞임)
(b) "이상적" - plaintext soft-MGI tree가 고른 hard feature=2, threshold=-0.5

이 둘로 계산한 child weight가 다르면, 그 weight로 child의 candidate score/gap이 얼마나
달라지는지까지 확인.
"""

from __future__ import annotations

import time

import numpy as np

from client_assisted.candidates import make_public_grid_candidates, make_small_public_threshold_grid
from client_assisted.dataset import load_scaled_dataset_subset, one_hot_encode
from fully_encrypted_mgi_stump import create_bootstrap_context, encrypt_dataset
from fully_encrypted_mgi_tree import find_best_split_weighted
from sigmoid_approx_coeffs import SIGMOID_APPROX_STEEPNESS

DATASET = "iris"
CANDIDATE_COUNT = 3
ROOT_SHARPEN = 19  # 지난 depth=3 실행에서 root에 쓴 값


def true_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(dataset_name=DATASET, test_size=30)
y_train_one_hot = one_hot_encode(y_train, n_classes=len(class_names))
candidates = make_public_grid_candidates(
    n_features=X_train.shape[1], thresholds=make_small_public_threshold_grid(candidate_count=CANDIDATE_COUNT)
)
normalizer = float(X_train.shape[0] ** 2)
n_features = X_train.shape[1]

ctx = create_bootstrap_context(mode="gpu")
dataset = encrypt_dataset(ctx, X_train, y_train_one_hot)
root_weights = ctx.engine.encrypt([1.0] * dataset.n_samples, ctx.pk)

t0 = time.time()
selected_split = find_best_split_weighted(
    ctx, dataset, candidates, root_weights, normalizer, ROOT_SHARPEN, verbose=True
)
print(f"\n[root node] time={time.time()-t0:.1f}s", flush=True)

actual_threshold = float(ctx.engine.decrypt(selected_split.enc_threshold, ctx.sk)[0].real)
actual_feat_conf = np.array(
    [float(ctx.engine.decrypt(fm, ctx.sk)[0].real) for fm in selected_split.enc_feature_masks]
)
print(f"실제 root threshold={actual_threshold}")
print(f"실제 root feature_confidence(4개 전부)={actual_feat_conf}")

# --- 시나리오 (a): 실제(soft, 여러 feature 섞인) selected_feature 사용 ---
actual_selected_feature = (X_train * actual_feat_conf[None, :]).sum(axis=1)
actual_right = true_sigmoid(SIGMOID_APPROX_STEEPNESS * (actual_selected_feature - actual_threshold))
actual_left_weights = 1.0 - actual_right

# --- 시나리오 (b): plaintext 이상적(hard feature=2, threshold=-0.5) ---
ideal_right = true_sigmoid(SIGMOID_APPROX_STEEPNESS * (X_train[:, 2] - (-0.5)))
ideal_left_weights = 1.0 - ideal_right

print(f"\nleft weight 차이(실제 vs 이상적): 평균 절대 오차={np.mean(np.abs(actual_left_weights-ideal_left_weights)):.4f}, "
      f"최대={np.max(np.abs(actual_left_weights-ideal_left_weights)):.4f}")

# --- 두 weight로 child(depth=1, left)의 candidate score/gap 각각 계산 ---
def child_scores(weights):
    scores = []
    for c in candidates:
        x = X_train[:, c.feature_idx]
        right = true_sigmoid(SIGMOID_APPROX_STEEPNESS * (x - c.threshold))
        left = 1.0 - right
        lw, rw = weights * left, weights * right
        lc = (lw[:, None] * y_train_one_hot).sum(axis=0)
        rc = (rw[:, None] * y_train_one_hot).sum(axis=0)
        score = (lc.sum() ** 2 - (lc**2).sum()) + (rc.sum() ** 2 - (rc**2).sum())
        scores.append(score)
    return np.array(scores)


scores_actual = child_scores(actual_left_weights)
scores_ideal = child_scores(ideal_left_weights)

print("\n=== child(왼쪽 자식) candidate 점수 비교 ===")
for i, c in enumerate(candidates):
    print(f"  candidate{i} feat={c.feature_idx} thr={c.threshold:+.3f} | 실제 weight 기준 score={scores_actual[i]:10.2f} | 이상적 weight 기준 score={scores_ideal[i]:10.2f}")

best_actual = int(np.argmin(scores_actual))
best_ideal = int(np.argmin(scores_ideal))
sorted_actual = np.sort(scores_actual)
sorted_ideal = np.sort(scores_ideal)
gap_actual = (sorted_actual[1] - sorted_actual[0]) / normalizer
gap_ideal = (sorted_ideal[1] - sorted_ideal[0]) / normalizer

print(f"\n실제 weight 기준 best candidate = {best_actual}({candidates[best_actual]}), gap={gap_actual:.6f}")
print(f"이상적 weight 기준 best candidate = {best_ideal}({candidates[best_ideal]}), gap={gap_ideal:.6f}")
print(f"\n=> best candidate가 같은가: {best_actual == best_ideal}")
print(f"=> gap이 얼마나 다른가: 실제/이상적 비율 = {gap_actual/gap_ideal if gap_ideal > 0 else float('nan'):.3f}x")
print("(공식으로 sharpen을 정할 때 '이상적' gap을 썼는데, 실제 gap이 이것보다 훨씬 작다면")
print(" 그게 바로 지난 실행에서 5~7번 노드가 sharpen을 올려도 안 좋아졌던 이유일 수 있음)")
