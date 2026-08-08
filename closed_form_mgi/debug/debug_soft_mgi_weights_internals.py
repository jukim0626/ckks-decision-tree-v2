"""soft_mgi_weights 내부 로직을 풀어서 반복마다 decrypt (어디서 깨지는지 특정)."""
from __future__ import annotations

import numpy as np

from client_assisted.dataset import load_scaled_dataset_subset, one_hot_encode
from closed_form_mgi.exp_approx_coeffs import exp_neg_beta_coeffs
from closed_form_mgi.primitives import create_bootstrap_context, encrypt_dataset, ensure_level
from closed_form_mgi.train import encrypted_weighted_threshold, evaluate_closed_form_candidates_from_thresholds

X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset("iris", test_size=30)
n_classes = len(class_names)
n_features = X_train.shape[1]
n_samples = X_train.shape[0]
y_train_oh = one_hot_encode(y_train, n_classes=n_classes)
score_normalizer = float(n_samples**2)

ctx = create_bootstrap_context(mode="gpu")
train_dataset = encrypt_dataset(ctx, X_train, y_train_oh)
node_weights = ctx.engine.encrypt([1.0] * n_samples, ctx.pk)

def dec(ct, n=4):
    vals = [v.real for v in ctx.engine.decrypt(ct, ctx.sk)[:n]]
    return np.array(vals)

thresholds = [encrypted_weighted_threshold(ctx, train_dataset.enc_features[j], node_weights, n_samples) for j in range(n_features)]
packed_score, gates, n_pow2, _ = evaluate_closed_form_candidates_from_thresholds(ctx, train_dataset, node_weights, thresholds, score_normalizer)
print(f"packed_score = {np.round(dec(packed_score, n_pow2), 4)}  level={packed_score.level}")

beta = 10.0
n_valid = n_features
degree = 20
reciprocal_iterations = 40

normalized_score = ctx.engine.multiply(packed_score, 1.0)
print(f"normalized_score = {np.round(dec(normalized_score, n_pow2),4)}  level={normalized_score.level}")

exp_coeffs = exp_neg_beta_coeffs(beta, degree=degree)
exp_val = ctx.engine.evaluate_polynomial(normalized_score, exp_coeffs, ctx.rlk)
print(f"exp_val = {np.round(dec(exp_val, n_pow2),6)}  (expect exp(-10*score)) level={exp_val.level}")
expected_exp = np.exp(-beta * dec(packed_score, n_pow2))
print(f"expected_exp = {np.round(expected_exp,6)}")

if n_pow2 > n_valid:
    valid_mask = np.array([1.0] * n_valid + [0.0] * (n_pow2 - n_valid))
    exp_val = ctx.engine.multiply(exp_val, valid_mask)

exp_val_for_sum = ctx.engine.intt(exp_val)
denom = ctx.engine.sum(exp_val_for_sum, ctx.rotation_key)
print(f"denom = {dec(denom,1)}  (expect {expected_exp.sum():.6f})  level={denom.level}")

y0 = 1.0 / n_valid
z = ctx.engine.multiply(denom, y0)
w = ctx.engine.multiply(exp_val, y0)
print(f"z0 = {dec(z,1)}  (expect {expected_exp.sum()*y0:.6f})  level={z.level}")
print(f"w0 = {np.round(dec(w,n_pow2),6)}  level={w.level}")

for i in range(reciprocal_iterations):
    z = ensure_level(ctx, z)
    w = ensure_level(ctx, w)
    two_minus_z = ctx.engine.subtract(2.0, z)
    z_new = ctx.engine.multiply(z, two_minus_z, ctx.rlk)
    w = ctx.engine.multiply(w, two_minus_z, ctx.rlk)
    z = z_new
    if i < 10 or i % 5 == 0 or i == reciprocal_iterations - 1:
        zval = dec(z, 1)[0]
        wval = dec(w, n_pow2)
        print(f"  iter {i:2d}: z={zval:.6f} (level={z.level})  w={np.round(wval,6)} (level={w.level})")
