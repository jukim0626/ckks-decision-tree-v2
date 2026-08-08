"""fully_encrypted_mgi_tree.py로 iris depth=3 tree를 학습하고, client-assisted depth=3
결과 및 plaintext soft-MGI 재귀 tree와 비교."""

from __future__ import annotations

import sys
import time

import numpy as np

from client_assisted import (
    create_context,
    encrypt_dataset as ca_encrypt_dataset,
    load_scaled_dataset_subset,
    make_public_grid_candidates,
    make_small_public_threshold_grid,
    one_hot_encode,
    train_client_assisted_fixed_depth_tree,
)
from fully_encrypted_mgi_stump import create_bootstrap_context, encrypt_dataset
from fully_encrypted_mgi_tree import train_fully_encrypted_mgi_tree
from sharpen_iterations_formula import required_sharpen_iterations
from sigmoid_approx_coeffs import SIGMOID_APPROX_STEEPNESS

DATASET = sys.argv[1] if len(sys.argv) > 1 else "iris"
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 3
CANDIDATE_COUNT = 3
TARGET_CONFIDENCE = 0.95
MIN_SHARPEN = 12  # 공식이 이보다 낮은 값을 추천해도 최소 이만큼은 씀 (안전 하한)


def true_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def plaintext_soft_mgi_tree(X, y_one_hot, candidates, depth, n_features, node_gaps=None):
    """검증용: plaintext soft-MGI 기준 재귀 tree (fully-encrypted와 같은 기준 사용).

    node_gaps가 주어지면 노드별(pre-order) (best_score, second_best_score, normalized_gap)를
    채워넣는다 - required_sharpen_iterations를 노드마다 계산하기 위한 재료."""
    n_samples = X.shape[0]
    normalizer = float(n_samples**2)

    def build(weights, cur_depth):
        if cur_depth == depth:
            return {"leaf_counts": (weights[:, None] * y_one_hot).sum(axis=0)}
        scores = []
        for idx, c in enumerate(candidates):
            x = X[:, c.feature_idx]
            right = true_sigmoid(SIGMOID_APPROX_STEEPNESS * (x - c.threshold))
            left = 1.0 - right
            lw, rw = weights * left, weights * right
            lc = (lw[:, None] * y_one_hot).sum(axis=0)
            rc = (rw[:, None] * y_one_hot).sum(axis=0)
            score = (lc.sum() ** 2 - (lc**2).sum()) + (rc.sum() ** 2 - (rc**2).sum())
            scores.append(score)
        sorted_scores = sorted(scores)
        best_score = sorted_scores[0]
        second_best = next((s for s in sorted_scores[1:] if not np.isclose(s, best_score, atol=1e-6)), sorted_scores[-1])
        gap = (second_best - best_score) / normalizer
        best_idx = int(np.argmin(scores))
        if node_gaps is not None:
            node_gaps.append(gap)
        c = candidates[best_idx]
        x = X[:, c.feature_idx]
        right = true_sigmoid(SIGMOID_APPROX_STEEPNESS * (x - c.threshold))
        left = 1.0 - right
        node = {"feature": c.feature_idx, "threshold": c.threshold, "score": best_score, "gap": gap}
        node["left"] = build(weights * left, cur_depth + 1)
        node["right"] = build(weights * right, cur_depth + 1)
        return node

    return build(np.ones(n_samples), 0)


def flatten_splits(node, depth, cur_depth=0, out=None):
    if out is None:
        out = []
    if cur_depth == depth:
        return out
    out.append((node["feature"], node["threshold"]))
    flatten_splits(node["left"], depth, cur_depth + 1, out)
    flatten_splits(node["right"], depth, cur_depth + 1, out)
    return out


X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(dataset_name=DATASET, test_size=30)
y_train_one_hot = one_hot_encode(y_train, n_classes=len(class_names))
candidates = make_public_grid_candidates(
    n_features=X_train.shape[1], thresholds=make_small_public_threshold_grid(candidate_count=CANDIDATE_COUNT)
)
print(f"[setup] dataset={DATASET} depth={DEPTH} candidates={len(candidates)} n_samples={X_train.shape[0]}", flush=True)

node_gaps: list = []
plain_tree = plaintext_soft_mgi_tree(X_train, y_train_one_hot, candidates, DEPTH, X_train.shape[1], node_gaps=node_gaps)
plain_splits = flatten_splits(plain_tree, DEPTH)
print(f"[plaintext soft-MGI tree] splits(pre-order)={plain_splits}", flush=True)

sharpen_per_node = [
    max(MIN_SHARPEN, required_sharpen_iterations(gap, TARGET_CONFIDENCE)) for gap in node_gaps
]
print(f"[node gaps(pre-order, plaintext 기준 계산)]={[round(g, 6) for g in node_gaps]}", flush=True)
print(f"[공식 추천 sharpen_iterations(node별, conf={TARGET_CONFIDENCE}, 최소 {MIN_SHARPEN})]={sharpen_per_node}", flush=True)

# --- client-assisted baseline ---
ctx1 = create_context(mode="gpu", max_level=15 if DEPTH >= 3 else 25)
dataset1 = ca_encrypt_dataset(ctx1, X_train, y_train_one_hot)
t0 = time.time()
model1, selections1 = train_client_assisted_fixed_depth_tree(ctx1, dataset1, candidates, depth=DEPTH, verbose=False)
time1 = time.time() - t0
ca_splits = [(s.feature_idx, s.threshold) for s in selections1.selections]
print(f"\n[client-assisted] time={time1:.2f}s | splits={ca_splits}", flush=True)
del ctx1, dataset1

# --- fully-encrypted MGI tree ---
ctx2 = create_bootstrap_context(mode="gpu")
dataset2 = encrypt_dataset(ctx2, X_train, y_train_one_hot)
t0 = time.time()
node_splits2, leaf_counts2 = train_fully_encrypted_mgi_tree(
    ctx2, dataset2, candidates, depth=DEPTH, sharpen_iterations=sharpen_per_node, verbose=True
)
time2 = time.time() - t0

decoded_splits = []
for split in node_splits2:
    thr = float(ctx2.engine.decrypt(split.enc_threshold, ctx2.sk)[0].real)
    feat_conf = [float(ctx2.engine.decrypt(fm, ctx2.sk)[0].real) for fm in split.enc_feature_masks]
    feat = max(range(len(feat_conf)), key=lambda i: feat_conf[i])
    decoded_splits.append((feat, thr, feat_conf[feat]))

decoded_leaf_counts = []
for lc in leaf_counts2:
    decoded_leaf_counts.append([float(ctx2.engine.decrypt(c, ctx2.sk)[0].real) for c in lc])

print(f"\n[fully-encrypted MGI tree] time={time2:.2f}s", flush=True)
print(f"splits(pre-order, feature/threshold/confidence)={decoded_splits}")
print(f"leaf_counts={decoded_leaf_counts}")

print(f"\n=== 비교 ===")
print(f"plaintext soft-MGI: {plain_splits}")
print(f"client-assisted:    {ca_splits}")
print(f"fully-encrypted:     {[(f, round(t, 4)) for f, t, c in decoded_splits]}")
print(f"시간: client-assisted={time1:.2f}s | fully-encrypted={time2:.2f}s | 배율={time2/time1:.2f}x")


def predict_soft(x, splits, leaf_counts, depth):
    weights = [1.0]
    idx = 0
    for _ in range(depth):
        new_weights = []
        for w in weights:
            feat, thr, _ = splits[idx]
            idx += 1
            right = true_sigmoid(SIGMOID_APPROX_STEEPNESS * (x[feat] - thr))
            left = 1.0 - right
            new_weights.append(w * left)
            new_weights.append(w * right)
        weights = new_weights
    class_scores = np.zeros(len(leaf_counts[0]))
    for w, lc in zip(weights, leaf_counts):
        class_scores += w * np.array(lc)
    return int(np.argmax(class_scores))


preds = np.array([predict_soft(x, decoded_splits, decoded_leaf_counts, DEPTH) for x in X_test])
test_acc = float(np.mean(preds == y_test))
print(f"\n[fully-encrypted tree] test accuracy (soft traversal, decrypted split/leaf 기준 재구성) = {test_acc * 100:.2f}%")
print(f"predicted class distribution: {np.bincount(preds, minlength=len(class_names))}")
print(f"true class distribution:      {np.bincount(y_test, minlength=len(class_names))}")
