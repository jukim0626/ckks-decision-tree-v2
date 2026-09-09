"""Phase 2/3/4의 새 gradient 공식(leaf_softmax=False, attention_softmax=False+regularizer,
native low-degree gate derivative)이 실제로 맞는지 finite-difference로 검증하는 스크립트.
GPU/CKKS 전혀 안 쓰고 numpy만 사용 - opt/tree_ops.py로 GPU 실험을 돌리기 전에 반드시
통과해야 하는 게이트 ([[feedback-verify-before-answering]] "실제로 검증할 것" 원칙)."""

from __future__ import annotations

import numpy as np

from experiments.gradient_soft_tree.opt.config import TreeConfig
from experiments.gradient_soft_tree.opt.reference_variants import _gate_and_deriv, regularizer_value
from experiments.gradient_soft_tree.depth1_reference import softmax, softmax_backward


def loss_and_analytic_grad(X, y_onehot, depth, alpha, threshold, leaf_logits, config: TreeConfig):
    n_samples, n_features = X.shape
    n_classes = y_onehot.shape[1]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth

    node_gate = [None] * n_internal
    node_gate_j = [None] * n_internal
    node_diff = [None] * n_internal
    node_w = [None] * n_internal
    node_parent_prob = [None] * n_internal

    current_level_probs = [np.ones(n_samples)]
    node_idx = 0
    for _level in range(depth):
        next_level_probs = []
        for parent_prob_arr in current_level_probs:
            i = node_idx
            w = softmax(alpha[i]) if config.attention_softmax else alpha[i]
            diff = X - threshold[i]
            gate_j, _ = _gate_and_deriv(diff, config)
            gate = gate_j @ w
            node_gate[i], node_gate_j[i], node_diff[i], node_w[i] = gate, gate_j, diff, w
            node_parent_prob[i] = parent_prob_arr
            next_level_probs.append(parent_prob_arr * (1.0 - gate))
            next_level_probs.append(parent_prob_arr * gate)
            node_idx += 1
        current_level_probs = next_level_probs
    leaf_probs = current_level_probs

    leafdist = [softmax(leaf_logits[l]) for l in range(n_leaves)] if config.leaf_softmax else list(leaf_logits)
    y_hat = sum(np.outer(leaf_probs[l], leafdist[l]) for l in range(n_leaves))
    # dL_dyhat = (2/n_samples)*(y_hat-y) 공식과 일치하는 loss: sum over classes, mean over
    # samples만 (즉 sum()/n_samples - np.mean()처럼 n_classes로 또 나누면 안 됨).
    loss = float(np.sum((y_hat - y_onehot) ** 2) / n_samples)
    for i in range(n_internal):
        loss += regularizer_value(alpha[i], config)
    if not config.leaf_softmax and config.lambda_leaf != 0.0:
        for l in range(n_leaves):
            loss += config.lambda_leaf * float(np.sum(leaf_logits[l] ** 2))
    dL_dyhat = (2.0 / n_samples) * (y_hat - y_onehot)

    current_g = []
    dL_dleaf_logits = [None] * n_leaves
    for l in range(n_leaves):
        dL_ddist_l = dL_dyhat.T @ leaf_probs[l]
        dL_dleaf_logits[l] = softmax_backward(leafdist[l], dL_ddist_l) if config.leaf_softmax else dL_ddist_l
        if not config.leaf_softmax and config.lambda_leaf != 0.0:
            dL_dleaf_logits[l] = dL_dleaf_logits[l] + 2.0 * config.lambda_leaf * leaf_logits[l]
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
            gate, gate_j, diff, w = node_gate[i], node_gate_j[i], node_diff[i], node_w[i]
            p_i = node_parent_prob[i]
            _, surrogate = _gate_and_deriv(diff, config)

            dL_dgate_i = p_i * (g_right - g_left)
            next_g[idx] = g_left * (1.0 - gate) + g_right * gate

            # steepness=8은 gate_j*(1-gate_j) 로지스틱 surrogate를 쓸 때만 필요한 외부
            # chain-rule 상수(gate_j=sigmoid(8*diff) 가정). use_true_gate_derivative=True면
            # surrogate가 이미 P'(diff) 그 자체라 d(diff)/d(theta)=-1만 곱하면 된다(diff에
            # steepness가 안 곱해져 있으므로 8을 또 곱하면 안 됨 - 처음엔 이 버그로 finite-diff
            # 체크가 실패했다, grad_check.py 실행 로그 참고).
            theta_chain_const = -1.0 if config.use_true_gate_derivative else -8.0
            dL_dthreshold[i] = theta_chain_const * w * (dL_dgate_i[:, None] * surrogate).sum(axis=0)
            if config.attention_softmax:
                term = dL_dgate_i[:, None] * (gate_j - gate[:, None])
                dL_dalpha[i] = w * term.sum(axis=0)
            else:
                term = dL_dgate_i[:, None] * gate_j
                from experiments.gradient_soft_tree.opt.reference_variants import _regularizer_grad
                dL_dalpha[i] = term.sum(axis=0) + _regularizer_grad(alpha[i], config)
        current_g = next_g

    return loss, np.array(dL_dalpha), np.array(dL_dthreshold), np.array(dL_dleaf_logits)


def finite_diff_check(config: TreeConfig, depth=2, n_features=3, n_classes=2, n_samples=15, eps=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, size=(n_samples, n_features))
    y = rng.integers(0, n_classes, size=n_samples)
    y_onehot = np.eye(n_classes)[y]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth
    alpha = rng.normal(0, 0.3, size=(n_internal, n_features))
    threshold = rng.normal(0, 0.3, size=(n_internal, n_features))
    leaf_logits = rng.normal(0, 0.3, size=(n_leaves, n_classes))

    loss0, g_alpha, g_thresh, g_leaf = loss_and_analytic_grad(X, y_onehot, depth, alpha, threshold, leaf_logits, config)

    def loss_at(a, t, l):
        loss, *_ = loss_and_analytic_grad(X, y_onehot, depth, a, t, l, config)
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
    print(f"[{config.name}] max relative grad error = {max_rel_err:.2e}")
    return max_rel_err


if __name__ == "__main__":
    configs = [
        TreeConfig(name="baseline"),
        TreeConfig(name="no_leaf_softmax", leaf_softmax=False),
        TreeConfig(name="no_attention_softmax_Rsum", attention_softmax=False, lambda_sum=0.01),
        TreeConfig(name="no_attention_softmax_Rsum_Rbinary", attention_softmax=False, lambda_sum=0.01, lambda_binary=0.01),
        TreeConfig(name="no_leaf_no_attention", leaf_softmax=False, attention_softmax=False, lambda_sum=0.01),
        TreeConfig(name="no_leaf_softmax_with_leaf_l2", leaf_softmax=False, lambda_leaf=0.05),
        TreeConfig(
            name="no_leaf_no_attention_with_leaf_l2", leaf_softmax=False, attention_softmax=False,
            lambda_sum=0.01, lambda_leaf=0.05,
        ),
        TreeConfig(name="gate_degree5_true_deriv", gate_degree=5, use_true_gate_derivative=True),
        TreeConfig(name="gate_degree3_true_deriv", gate_degree=3, use_true_gate_derivative=True),
        TreeConfig(name="gate_degree15_true_deriv", gate_degree=15, use_true_gate_derivative=True),
    ]
    failed = []
    for cfg in configs:
        err = finite_diff_check(cfg)
        if err > 1e-3:
            failed.append((cfg.name, err))
    if failed:
        print("\nFAILED:", failed)
    else:
        print("\nAll gradient checks passed (rel err < 1e-3).")
