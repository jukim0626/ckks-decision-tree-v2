"""depthN_reference.train_depthN을 TreeConfig로 파라미터화한 plaintext(numpy) 버전.

opt/tree_ops.py의 CKKS 구현이 옳은지 GPU 없이 먼저 검증하는 용도(이 프로젝트의 검증된
관행 - closed_form_mgi 세션들에서 "값비싼 GPU 실험 전에 저비용 근사 분석으로 가설을
좁히는 게 효율적" [[project-ckks-decision-tree]]) + Phase 1/3의 plaintext sweep에도 재사용.

TreeConfig() 기본값으로 호출하면 depthN_reference.train_depthN과 정확히 같은 궤적을
내야 한다 (opt/validate.py가 이걸로 baseline 정합성을 확인한다)."""

from __future__ import annotations

import numpy as np

from experiments.gradient_soft_tree.depth1_reference import softmax, softmax_backward
from experiments.gradient_soft_tree.opt.config import TreeConfig
from experiments.gradient_soft_tree.opt.poly_gate import gate_coeffs, gate_derivative_coeffs

STEEPNESS = 8.0


def _gate_and_deriv(diff: np.ndarray, config: TreeConfig):
    if config.gate_degree == 15 and not config.use_true_gate_derivative:
        # baseline: 실제 sigmoid(steepness*diff) (CKKS의 degree-15 Chebyshev 근사에 대한
        # "이론적 상한" 기준 - depthN_reference.py와 동일)
        gate = 1.0 / (1.0 + np.exp(-STEEPNESS * diff))
        deriv = gate * (1.0 - gate)
        return gate, deriv
    coeffs = gate_coeffs(config.gate_degree)
    gate = np.polynomial.polynomial.polyval(diff, coeffs)
    if config.use_true_gate_derivative:
        deriv = np.polynomial.polynomial.polyval(diff, gate_derivative_coeffs(config.gate_degree))
    else:
        deriv = gate * (1.0 - gate)
    return gate, deriv


def _regularizer_grad(a_row: np.ndarray, config: TreeConfig) -> np.ndarray:
    grad = np.zeros_like(a_row)
    if config.lambda_sum != 0.0:
        s = a_row.sum()
        grad = grad + 2.0 * config.lambda_sum * (s - 1.0)
    if config.lambda_binary != 0.0:
        grad = grad + 2.0 * config.lambda_binary * a_row * (1 - a_row) * (1 - 2 * a_row)
    return grad


def regularizer_value(a_row: np.ndarray, config: TreeConfig) -> float:
    val = 0.0
    if config.lambda_sum != 0.0:
        val += config.lambda_sum * (a_row.sum() - 1.0) ** 2
    if config.lambda_binary != 0.0:
        val += config.lambda_binary * float(np.sum(a_row**2 * (1 - a_row) ** 2))
    return val


def train_depthN_variant(
    X: np.ndarray, y_onehot: np.ndarray, depth: int, config: TreeConfig,
    lr: float = 2.0, epochs: int = 15, seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    n_classes = y_onehot.shape[1]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth

    alpha = rng.normal(0, 0.1, size=(n_internal, n_features))
    threshold = rng.normal(0, 0.1, size=(n_internal, n_features))
    leaf_logits = rng.normal(0, 0.1, size=(n_leaves, n_classes))

    for _ in range(epochs):
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

        if config.leaf_softmax:
            leafdist = [softmax(leaf_logits[l]) for l in range(n_leaves)]
        else:
            leafdist = [leaf_logits[l] for l in range(n_leaves)]
        y_hat = sum(np.outer(leaf_probs[l], leafdist[l]) for l in range(n_leaves))
        dL_dyhat = (2.0 / n_samples) * (y_hat - y_onehot)

        current_g = []
        dL_dleaf_logits = [None] * n_leaves
        for l in range(n_leaves):
            dL_ddist_l = dL_dyhat.T @ leaf_probs[l]
            if config.leaf_softmax:
                dL_dleaf_logits[l] = softmax_backward(leafdist[l], dL_ddist_l)
            else:
                dL_dleaf_logits[l] = dL_ddist_l
                if config.lambda_leaf != 0.0:
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

                # grad_check.py에서 실측된 버그 수정과 동일: use_true_gate_derivative=True면
                # surrogate가 이미 P'(diff)라 steepness를 또 곱하면 안 되고 chain rule
                # 상수는 d(diff)/d(theta)=-1 뿐이다.
                theta_chain_const = -1.0 if config.use_true_gate_derivative else -STEEPNESS
                dL_dthreshold[i] = theta_chain_const * w * (dL_dgate_i[:, None] * surrogate).sum(axis=0)

                if config.attention_softmax:
                    term = dL_dgate_i[:, None] * (gate_j - gate[:, None])
                    dL_dalpha[i] = w * term.sum(axis=0)
                else:
                    term = dL_dgate_i[:, None] * gate_j
                    dL_dalpha[i] = term.sum(axis=0) + _regularizer_grad(alpha[i], config)
                current_g = current_g  # (no-op, keeps structure readable)
            current_g = next_g

        alpha -= lr * np.array(dL_dalpha)
        threshold -= lr * np.array(dL_dthreshold)
        leaf_logits -= lr * np.array(dL_dleaf_logits)

    return {
        "alpha": alpha, "threshold": threshold, "leaf_logits": leaf_logits,
        "steepness": STEEPNESS, "depth": depth, "config": config,
    }


def predict_variant(X: np.ndarray, params: dict) -> np.ndarray:
    config: TreeConfig = params["config"]
    depth = params["depth"]
    n_samples = X.shape[0]
    current_level_probs = [np.ones(n_samples)]
    node_idx = 0
    for _level in range(depth):
        next_level_probs = []
        for parent_prob_arr in current_level_probs:
            w = softmax(params["alpha"][node_idx]) if config.attention_softmax else params["alpha"][node_idx]
            diff = X - params["threshold"][node_idx]
            gate_j, _ = _gate_and_deriv(diff, config)
            gate = gate_j @ w
            next_level_probs.append(parent_prob_arr * (1.0 - gate))
            next_level_probs.append(parent_prob_arr * gate)
            node_idx += 1
        current_level_probs = next_level_probs
    leaf_probs = current_level_probs
    if config.leaf_softmax:
        leafdist = [softmax(params["leaf_logits"][l]) for l in range(len(leaf_probs))]
    else:
        leafdist = [params["leaf_logits"][l] for l in range(len(leaf_probs))]
    y_hat = sum(np.outer(leaf_probs[l], leafdist[l]) for l in range(len(leaf_probs)))
    return y_hat.argmax(axis=1)
