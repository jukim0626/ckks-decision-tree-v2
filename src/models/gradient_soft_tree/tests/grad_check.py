"""baseline/packed가 실제로 쓰는 gradient 공식(steepness=8 sigmoid gate, softmax attention,
softmax leaf distribution, vanilla GD)이 맞는지 finite-difference로 검증하는 test.
GPU/CKKS 전혀 안 쓰고 numpy만 사용 - 실제 CKKS 구현을 GPU로 돌리기 전에 반드시 통과해야
하는 게이트.

2026-09-22: opt 계보 제거와 함께 TreeConfig로 여러 ablation(leaf_softmax 끄기, low-degree
gate, attention softmax 제거 등)을 스윕하던 걸 걷어내고, 지금 실제로 쓰는 표준 공식
하나만 검증하도록 단순화했다(그 ablation들은 현재 지원 범위 밖). 검증하는 공식 자체는
`opt/reference_variants.py`가 `TreeConfig(name="baseline")` 기본값으로 계산하던 것과
100% 동일 - gate_degree=15/use_true_gate_derivative=False일 때 `_gate_and_deriv`가
`gate=sigmoid(8*diff)`, `deriv=gate*(1-gate)`를 반환하던 것과 같다."""

from __future__ import annotations

import numpy as np

from models.gradient_soft_tree.plaintext_softmax import softmax, softmax_backward

STEEPNESS = 8.0


def loss_and_analytic_grad(X, y_onehot, depth, alpha, threshold, leaf_logits):
    n_samples, n_features = X.shape
    n_classes = y_onehot.shape[1]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth

    node_gate = [None] * n_internal
    node_gate_j = [None] * n_internal
    node_w = [None] * n_internal
    node_parent_prob = [None] * n_internal

    current_level_probs = [np.ones(n_samples)]
    node_idx = 0
    for _level in range(depth):
        next_level_probs = []
        for parent_prob_arr in current_level_probs:
            i = node_idx
            w = softmax(alpha[i])
            diff = X - threshold[i]
            gate_j = 1.0 / (1.0 + np.exp(-STEEPNESS * diff))
            gate = gate_j @ w
            node_gate[i], node_gate_j[i], node_w[i] = gate, gate_j, w
            node_parent_prob[i] = parent_prob_arr
            next_level_probs.append(parent_prob_arr * (1.0 - gate))
            next_level_probs.append(parent_prob_arr * gate)
            node_idx += 1
        current_level_probs = next_level_probs
    leaf_probs = current_level_probs

    leafdist = [softmax(leaf_logits[l]) for l in range(n_leaves)]
    y_hat = sum(np.outer(leaf_probs[l], leafdist[l]) for l in range(n_leaves))
    # dL_dyhat = (2/n_samples)*(y_hat-y) 공식과 일치하는 loss: sum over classes, mean over
    # samples만 (즉 sum()/n_samples - np.mean()처럼 n_classes로 또 나누면 안 됨).
    loss = float(np.sum((y_hat - y_onehot) ** 2) / n_samples)
    dL_dyhat = (2.0 / n_samples) * (y_hat - y_onehot)

    current_g = []
    dL_dleaf_logits = [None] * n_leaves
    for l in range(n_leaves):
        dL_ddist_l = dL_dyhat.T @ leaf_probs[l]
        dL_dleaf_logits[l] = softmax_backward(leafdist[l], dL_ddist_l)
        current_g.append(dL_dyhat @ leafdist[l])

    dL_dalpha = [None] * n_internal
    dL_dthreshold = [None] * n_internal
    for level in reversed(range(depth)):
        start = (1 << level) - 1
        count = 1 << level
        next_g = [None] * count
        for idx in range(count):
            i = start + idx
            g_left, g_right = current_g[2 * idx], current_g[2 * idx + 1]
            gate, gate_j, w = node_gate[i], node_gate_j[i], node_w[i]
            p_i = node_parent_prob[i]
            surrogate = gate_j * (1.0 - gate_j)

            dL_dgate_i = p_i * (g_right - g_left)
            next_g[idx] = g_left * (1.0 - gate) + g_right * gate

            dL_dthreshold[i] = -STEEPNESS * w * (dL_dgate_i[:, None] * surrogate).sum(axis=0)
            term = dL_dgate_i[:, None] * (gate_j - gate[:, None])
            dL_dalpha[i] = w * term.sum(axis=0)
        current_g = next_g

    return loss, np.array(dL_dalpha), np.array(dL_dthreshold), np.array(dL_dleaf_logits)


def finite_diff_check(depth=2, n_features=3, n_classes=2, n_samples=15, eps=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, size=(n_samples, n_features))
    y = rng.integers(0, n_classes, size=n_samples)
    y_onehot = np.eye(n_classes)[y]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth
    alpha = rng.normal(0, 0.3, size=(n_internal, n_features))
    threshold = rng.normal(0, 0.3, size=(n_internal, n_features))
    leaf_logits = rng.normal(0, 0.3, size=(n_leaves, n_classes))

    loss0, g_alpha, g_thresh, g_leaf = loss_and_analytic_grad(X, y_onehot, depth, alpha, threshold, leaf_logits)

    def loss_at(a, t, l):
        loss, *_ = loss_and_analytic_grad(X, y_onehot, depth, a, t, l)
        return loss

    max_rel_err = 0.0
    for name, arr, analytic in (("alpha", alpha, g_alpha), ("threshold", threshold, g_thresh), ("leaf_logits", leaf_logits, g_leaf)):
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            orig = arr[idx]
            arr[idx] = orig + eps
            lp = loss_at(alpha, threshold, leaf_logits)
            arr[idx] = orig - eps
            lm = loss_at(alpha, threshold, leaf_logits)
            arr[idx] = orig
            fd = (lp - lm) / (2 * eps)
            an = analytic[idx]
            denom = max(abs(fd), abs(an), 1e-8)
            rel = abs(fd - an) / denom
            max_rel_err = max(max_rel_err, rel)
    print(f"[depth={depth} n_features={n_features} n_classes={n_classes}] max relative grad error = {max_rel_err:.2e}")
    return max_rel_err


def test_grad_check() -> None:
    failed = []
    for depth, n_features, n_classes, seed in [(1, 4, 3, 0), (2, 3, 2, 0), (3, 4, 3, 1)]:
        err = finite_diff_check(depth=depth, n_features=n_features, n_classes=n_classes, seed=seed)
        if err > 1e-3:
            failed.append((depth, n_features, n_classes, err))
    if failed:
        print("\nFAILED:", failed)
        raise AssertionError(f"gradient check 실패: {failed}")
    print("\nAll gradient checks passed (rel err < 1e-3).")


if __name__ == "__main__":
    test_grad_check()
