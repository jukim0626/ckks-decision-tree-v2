"""encrypted_weighted_threshold 하나만 떼어내서 decrypt 검증 (root, feature=0)."""
from __future__ import annotations

import numpy as np

from client_assisted.dataset import load_scaled_dataset_subset, one_hot_encode
from closed_form_mgi.primitives import create_bootstrap_context, encrypt_dataset
from closed_form_mgi.train import encrypted_weighted_threshold

X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset("iris", test_size=30)
n_classes = len(class_names)
y_train_oh = one_hot_encode(y_train, n_classes=n_classes)

expected = float(X_train[:, 0].mean())
print(f"expected (plaintext weighted mean, root node_weight=1) = {expected:.4f}")

ctx = create_bootstrap_context(mode="gpu")
train_dataset = encrypt_dataset(ctx, X_train, y_train_oh)
node_weights = ctx.engine.encrypt([1.0] * X_train.shape[0], ctx.pk)

for iters in [5, 10, 20, 30]:
    t = encrypted_weighted_threshold(ctx, train_dataset.enc_features[0], node_weights, X_train.shape[0], iterations=iters)
    val = float(ctx.engine.decrypt(t, ctx.sk)[0].real)
    print(f"iterations={iters:3d} -> decrypted threshold = {val:.4f}  (level={t.level})")
