# CKKS Encrypted Decision Tree Training

Fully homomorphic (CKKS) training and inference of soft decision trees — the server never
sees plaintext data, model parameters, or intermediate values. Both training and inference
run entirely on encrypted data.

## Why "soft" trees

CKKS supports addition and multiplication on encrypted values, but not comparison
(`x < threshold`). So instead of a hard split, every internal node routes samples with a
sigmoid gate, approximated by a Chebyshev polynomial:

```
right_prob = sigmoid(steepness * (x - threshold))
left_prob  = 1 - right_prob
```

A sample's probability of reaching a given leaf is the product of the gate values along the
root-to-leaf path. Training minimizes the leaf prediction loss end-to-end with gradient
descent — no Gini/impurity computation, no `argmin` over candidate splits, and no
client-side decryption step during training.

## Layout

```
src/
├── core/                        # primitives shared across all models
│   ├── ckks_engine.py            # CKKS engine/context creation, level (bootstrap) management
│   ├── approximation/            # polynomial approximations (sigmoid, exp)
│   ├── encrypted_ops/            # softmax, SIMD slot packing on ciphertexts
│   └── data/                     # dataset loading/encryption, process-boundary serialization
│
└── models/
    └── gradient_soft_tree/
        ├── gate.py                # shared gate computation (baseline + local_loss)
        ├── baseline/              # gradient-descent soft tree, verified for depth 1-3
        ├── opt/                   # bootstrap-count reduction experiments
        ├── packed/                # feature-axis SIMD packing (~26x speedup on a depth-3 tree)
        └── local_loss/            # per-level local-loss training (depth-independent backward depth)
```

## Setup

```bash
pip install desilofhe scikit-learn numpy torch
```

## Usage

All commands run from `src/`, since `core` and `models` are top-level packages there.

```bash
cd src

# baseline training
python -m models.gradient_soft_tree.baseline.train <dataset> <depth> <epochs> <lr> <seed> <level_preset>
# example
python -m models.gradient_soft_tree.baseline.train iris 3 35 2.0 0 17

# feature-axis SIMD packing
python -m models.gradient_soft_tree.packed.train <dataset> <depth> <epochs> <lr> <seed> <level_preset>
```

`<dataset>` is one of `iris`, `wine`, `breast_cancer`, `digits`, `diabetes`.

## Notes

- GPU-backed CKKS (via `desilofhe`); long training runs are typically detached with `nohup`
  since they can take minutes per epoch at higher depths.
- Earlier approaches (client-assisted decrypt-based splitting, closed-form MGI with
  encrypted argmin, oblique gates) are kept locally for reference but aren't part of this
  public tree, since the project settled on a fully end-to-end, comparison-free design.
