"""closed_form_mgi 파이프라인을 root 노드에서 단계별로 decrypt해서 plaintext 기댓값과 비교."""
from __future__ import annotations

import numpy as np

from client_assisted.dataset import load_scaled_dataset_subset, one_hot_encode
from closed_form_mgi.primitives import create_bootstrap_context, encrypt_dataset
from closed_form_mgi.soft_mgi import soft_mgi_weights
from closed_form_mgi.train import blended_gate_from_gates, encrypted_weighted_threshold, evaluate_closed_form_candidates_from_thresholds

X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset("iris", test_size=30)
n_classes = len(class_names)
n_features = X_train.shape[1]
n_samples = X_train.shape[0]
y_train_oh = one_hot_encode(y_train, n_classes=n_classes)
score_normalizer = float(n_samples**2)

ctx = create_bootstrap_context(mode="gpu")
train_dataset = encrypt_dataset(ctx, X_train, y_train_oh)
node_weights = ctx.engine.encrypt([1.0] * n_samples, ctx.pk)

def dec(ct, n=1):
    vals = [v.real for v in ctx.engine.decrypt(ct, ctx.sk)[:n]]
    return vals[0] if n == 1 else np.array(vals)

print("=== thresholds ===")
thresholds = []
for j in range(n_features):
    t = encrypted_weighted_threshold(ctx, train_dataset.enc_features[j], node_weights, n_samples)
    thresholds.append(t)
    print(f"  feat[{j}] threshold={dec(t):.4f}  level={t.level}")

print("\n=== candidates (gates/scores) ===")
packed_score, gates, n_pow2, n_features_ret = evaluate_closed_form_candidates_from_thresholds(
    ctx, train_dataset, node_weights, thresholds, score_normalizer
)
print(f"n_pow2={n_pow2}")
packed_score_dec = dec(packed_score, n_pow2)
print(f"packed_score(normalized, expect ~[0.51,1.0,0.0,0.0066] for iris root) = {np.round(packed_score_dec,4)}")
for j in range(n_features):
    g = dec(gates[j], 5)
    print(f"  gate[{j}] first5 = {np.round(g,4)}  level={gates[j].level}")

print(f"\npacked_score level={packed_score.level}")

# 가설 검증: score_normalizer=1.0(정확히 1을 곱함)이 문제인지 확인 -> 0.5로 바꿔서 테스트
half_score = ctx.engine.multiply(packed_score, 0.5)
for beta_try in [1.0, 10.0, 30.0]:
    print(f"\n=== soft_mgi_weights beta={beta_try}, score_normalizer=0.5(1.0 회피) ===")
    weights = soft_mgi_weights(ctx, half_score, n_features, n_pow2, 0.5, beta=beta_try)
    w_dec = dec(weights, n_pow2)
    print(f"weights = {np.round(w_dec,4)}  sum={w_dec[:n_features].sum():.4f}  level={weights.level}")

for beta_try in [1.0, 10.0, 30.0]:
    print(f"\n=== soft_mgi_weights beta={beta_try}, score_normalizer=1.0(원래 방식) ===")
    weights = soft_mgi_weights(ctx, packed_score, n_features, n_pow2, 1.0, beta=beta_try)
    w_dec = dec(weights, n_pow2)
    print(f"weights = {np.round(w_dec,4)}  sum={w_dec[:n_features].sum():.4f}  level={weights.level}")
