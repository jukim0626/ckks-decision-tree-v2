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
        │                          #   (+ instrumentation.py: op/bootstrap counters, GPU peak memory)
        ├── tests/                 # automated pass/fail checks (CPU-only, no GPU needed)
        └── experiments/           # manual diagnostics and legacy-session migration tools
```

Current research focus is **joint training on the `packed` path** (single leaf loss backpropagated
through the whole tree, vanilla gradient descent) - `baseline` is kept only as the plaintext/CKKS
comparison reference it has always been. Two earlier lineages (`local_loss`, a per-level local-loss
training variant, and `opt`, a set of bootstrap-count-reduction ablations) were explored and are no
longer supported; their code was removed rather than kept around unused. Both are still reachable in
git history (tag `pre-cleanup-packed-joint-optim`) if needed again.

## How the `packed` path works

All training samples of one feature live in a single ciphertext (one slot per sample). The `packed`
path additionally places every feature in its own block of one wide ciphertext
(feature `j` -> slots `[j*B, (j+1)*B)`, `B = next_pow2(n_samples)`), so that per node:

- **forward**: all `(x_j - threshold_j)` differences go through the sigmoid polynomial **once**
  instead of once per feature;
- **backward**: threshold/attention gradient reductions use a block-local rotate-and-add
  (`log2(B)` rotations for all features at once) instead of one full-width sum per feature;
- **softmax batching**: the attention softmax of all nodes on one tree level, and the class
  softmax of all leaves, are evaluated together in one blocked ciphertext (one exp polynomial
  and one Newton-Raphson reciprocal for the whole group);
- **public masks stay plaintext**: the sample mask depends only on `n_samples`, which is public,
  so it is multiplied as a plaintext (no relinearization).

Each training **iteration** is one **full-batch gradient-descent step** over the whole encrypted
training set (the CLI argument is still named `epochs`; 1 epoch = 1 iteration = 1 parameter update),
run in its own subprocess (setup / per-epoch / finalize workers) so GPU memory held by the CKKS
engine is fully released between steps.

## Results (depth 3, 30 iterations, `level_preset=17`, 80/20 split, seed 0)

| dataset | features | lr | first iteration | steady-state / iteration | train acc | test acc (encrypted inference) |
|---|---|---|---|---|---|---|
| iris | 4 | 2.0 | 428 s | ~275 s | 0.958 | 0.900 |
| wine | 13 | 6.0 | 723 s | ~291 s | 0.901 | 0.833 |
| breast_cancer | 30 | 2.0 | 1246 s | ~279 s | 0.908 | 0.947 |

Single NVIDIA GPU (24 GB). Test accuracy is measured by genuine encrypted inference
(`predict_packed`): the trained parameters are never decrypted, only the final class scores.

## Known limitations

- **Few optimization steps.** 30 iterations = 30 full-batch GD steps. Plaintext runs of the same model
  keep improving well beyond that (e.g. 300+ steps), so accuracy is currently step-limited.
- **Polynomial domain.** The sigmoid polynomial is fit on `[-2, 2]` and the softmax exp polynomial on
  `[-2.5, 2.5]`. Client-side scaling clips features to `[-1, 1]`, but nothing yet constrains the
  learned parameters to stay inside these intervals, which becomes an issue with longer or more
  aggressive training (finalize reports this as `domain(...)` diagnostics).
- **Parameter storage.** Thresholds are stored as one ciphertext per (node, feature), so very wide
  datasets or deeper trees run out of GPU memory; nodes on the same level are processed sequentially.

## Keys and evaluation

- **Key separation.** `session_dir/keys/` holds only public/evaluation keys (public, relinearization,
  rotation, conjugation, bootstrap). The secret key lives in `session_dir/client/sk.bin`, together with
  the client's scaler. Server-side workers (training iterations, the encrypted-inference forward pass)
  build their context without a secret key (`ctx.sk is None`), so any accidental decryption on the
  server path fails immediately.
- **Three accuracy levels.** `finalize` decrypts the trained parameters (client-side, verification only)
  and reports accuracy with the true sigmoid/exp (`[true-fn]`) and with the exact polynomials CKKS
  evaluates (`[poly]`). `max abs diff` against a true-function plaintext run measures polynomial
  approximation error plus CKKS noise; against a polynomial plaintext run it isolates CKKS noise.
  `predict_packed` gives the third level: genuine encrypted inference, which matches `[poly]`.

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
python -m models.gradient_soft_tree.packed.train <dataset> <depth> <epochs> <lr> <seed> <level_preset> [max_train] [test_size]
# example (test_size defaults to 0.2; max_train subsamples the training set, "none" = all)
python -m models.gradient_soft_tree.packed.train wine 3 30 6.0 0 17
# genuine encrypted inference on a trained session (never decrypts the model, only the final score)
python -m models.gradient_soft_tree.packed.predict_packed <session_dir>

# automated checks (models/gradient_soft_tree/tests/)
python -m models.gradient_soft_tree.tests.grad_check           # pure numpy, no CKKS
python -m models.gradient_soft_tree.tests.test_block_ops       # CPU-mode CKKS engine, no GPU needed
python -m models.gradient_soft_tree.tests.test_block_packed_softmax
python -m models.gradient_soft_tree.tests.test_packed_vs_plaintext <dataset> <depth> <epochs> <lr>  # GPU only
```

`<dataset>` is one of `iris`, `wine`, `breast_cancer`, `digits`, `diabetes`, `soybean`. Features are
min-max scaled to `[-1, 1]` on the client before encryption; the fitted scaler is saved with the
session so finalize/inference reuse it instead of refitting.

The packed layout requires `n_features * next_pow2(n_samples) <= 32768` slots; use `max_train` to
subsample wider datasets.

## Notes

- GPU-backed CKKS (via `desilofhe`); long training runs are typically detached with `nohup`
  since a depth-3 epoch takes several minutes.
- Earlier approaches (client-assisted decrypt-based splitting, closed-form MGI with
  encrypted argmin, oblique gates) are kept locally for reference but aren't part of this
  public tree, since the project settled on a fully end-to-end, comparison-free design.
