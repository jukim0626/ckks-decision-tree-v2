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
│   ├── ckks_engine.py             # CKKS engine/context creation, level (bootstrap) management
│   ├── runtime/                   # GPU/subprocess/session-dir plumbing shared by every train.py
│   ├── approximation/             # polynomial approximations (sigmoid, exp)
│   ├── encrypted_ops/             # softmax, SIMD block/slot packing on ciphertexts
│   └── data/                      # dataset loading/encryption, scaler persistence, process-boundary serialization
│
└── models/
    └── gradient_soft_tree/
        ├── gate.py                # baseline's axis-aligned attention-blend gate computation
        ├── params.py              # shared alpha/threshold/leaf_logits ciphertext I/O
        ├── plaintext_softmax.py   # plaintext (numpy) softmax used by reference implementations
        ├── baseline/              # gradient-descent soft tree, verified for depth 1-3 (comparison baseline)
        ├── packed/                # feature-axis SIMD packing (~26x speedup on a depth-3 tree) - current focus
        ├── tests/                 # automated pass/fail checks (CPU-only, no GPU needed)
        └── experiments/           # manual diagnostics and legacy-session migration tools
```

Current research focus is **joint training on the `packed` path** (single leaf loss backpropagated
through the whole tree, vanilla gradient descent) - `baseline` is kept only as the plaintext/CKKS
comparison reference it has always been. Two earlier lineages (`local_loss`, a per-level local-loss
training variant, and `opt`, a set of bootstrap-count-reduction ablations) were explored and are no
longer supported; their code was removed rather than kept around unused. Both are still reachable in
git history (tag `pre-cleanup-packed-joint-optim`) if needed again.

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

# feature-axis SIMD packing (current focus)
python -m models.gradient_soft_tree.packed.train <dataset> <depth> <epochs> <lr> <seed> <level_preset>
# genuine encrypted inference on a trained session (never decrypts the model, only the final score)
python -m models.gradient_soft_tree.packed.predict_packed <session_dir>

# automated checks (models/gradient_soft_tree/tests/)
python -m models.gradient_soft_tree.tests.grad_check           # pure numpy, no CKKS
python -m models.gradient_soft_tree.tests.test_block_ops       # CPU-mode CKKS engine, no GPU needed
python -m models.gradient_soft_tree.tests.test_block_packed_softmax
python -m models.gradient_soft_tree.tests.test_packed_vs_plaintext <dataset> <depth> <epochs> <lr>  # GPU only
```

`<dataset>` is one of `iris`, `wine`, `breast_cancer`, `digits`, `diabetes`.

## Notes

- GPU-backed CKKS (via `desilofhe`); long training runs are typically detached with `nohup`
  since they can take minutes per epoch at higher depths.
- Earlier approaches (client-assisted decrypt-based splitting, closed-form MGI with
  encrypted argmin, oblique gates) are kept locally for reference but aren't part of this
  public tree, since the project settled on a fully end-to-end, comparison-free design.
