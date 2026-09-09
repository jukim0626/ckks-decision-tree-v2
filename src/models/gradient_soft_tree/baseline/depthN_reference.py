"""depth1_reference.py를 임의 depth로 일반화한 CKKS-매칭 plaintext 기준 구현.

**일반화 핵심 아이디어**: depth=1은 root 하나가 곧 "마지막 internal level"이라 그 gate의
dL/dgate가 바로 두 leaf의 gradient 차이(`sum_c dL_dyhat_c*(leafdist_R-leafdist_L)`)였다.
depth>1에서는 이 관계가 "leaf" 대신 "자식 노드(혹은 자식 leaf)의 reach probability에 대한
gradient(g)"로 일반화된다:

  각 노드/leaf X의 "reach probability" p_X(그 샘플이 X에 도달할 확률)에 대해
  g_X := dL/dp_X 를 정의하면:
  - leaf l: g_l = sum_c dL_dyhat_c * leafdist_l_c  (depth=1의 dL_dgate 공식과 정확히 동일한
    형태 - depth=1은 root가 곧 "마지막 레벨"이라 이 값이 바로 dL_dgate였을 뿐)
  - internal node i(자식의 reach prob에 대한 g_left, g_right를 이미 알고 있다고 하면):
      dL_dgate_i = p_i * (g_right - g_left)   (p_i = 그 노드 자신의 incoming reach prob, root는 1)
      g_i        = g_left*(1-gate_i) + g_right*gate_i   (i의 부모가 필요로 하는 값)

  dL_dgate_i가 나오면 그 다음은 depth=1과 완전히 동일한 공식으로 그 노드의
  threshold/alpha gradient를 구한다. 즉 leaf에서 root 방향으로 레벨을 거슬러 올라가며
  이 공식을 반복 적용하는 게 전부 - depth=1의 backward가 "레벨 1개짜리" 특수 케이스였을 뿐,
  본질적으로 같은 계산이 레벨마다 반복된다. (CKKS 포팅 시 이 재귀 구조를 그대로 옮기면
  depth=1에서 이미 검증된 threshold/alpha 공식을 노드마다 재사용할 수 있다.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from models.gradient_soft_tree.baseline.depth1_reference import softmax, softmax_backward  # noqa: E402


def train_depthN(
    X: np.ndarray,
    y_onehot: np.ndarray,
    depth: int,
    steepness: float = 8.0,
    lr: float = 0.3,
    epochs: int = 300,
    seed: int = 0,
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
        # ---- forward: 레벨별로 내려가며 각 노드의 gate/incoming reach prob을 저장 ----
        node_gate = [None] * n_internal
        node_gate_j = [None] * n_internal
        node_w = [None] * n_internal
        node_parent_prob = [None] * n_internal  # 그 노드 자신의 incoming reach prob (root=1)

        current_level_probs = [np.ones(n_samples)]
        node_idx = 0
        for _level in range(depth):
            next_level_probs = []
            for parent_prob_arr in current_level_probs:
                i = node_idx
                w = softmax(alpha[i])
                gate_j = 1.0 / (1.0 + np.exp(-steepness * (X - threshold[i])))
                gate = gate_j @ w
                node_gate[i], node_gate_j[i], node_w[i] = gate, gate_j, w
                node_parent_prob[i] = parent_prob_arr
                next_level_probs.append(parent_prob_arr * (1.0 - gate))
                next_level_probs.append(parent_prob_arr * gate)
                node_idx += 1
            current_level_probs = next_level_probs
        leaf_probs = current_level_probs  # length n_leaves

        leafdist = [softmax(leaf_logits[l]) for l in range(n_leaves)]
        y_hat = sum(np.outer(leaf_probs[l], leafdist[l]) for l in range(n_leaves))
        dL_dyhat = (2.0 / n_samples) * (y_hat - y_onehot)

        # ---- leaf gradients + g(=dL/dp_leaf) ----
        current_g = []
        dL_dleaf_logits = [None] * n_leaves
        for l in range(n_leaves):
            dL_ddist_l = dL_dyhat.T @ leaf_probs[l]
            dL_dleaf_logits[l] = softmax_backward(leafdist[l], dL_ddist_l)
            current_g.append(dL_dyhat @ leafdist[l])

        # ---- 레벨을 거슬러 올라가며 노드 gradient 계산 ----
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

                dL_dgate_i = p_i * (g_right - g_left)
                next_g[idx] = g_left * (1.0 - gate) + g_right * gate

                surrogate = gate_j * (1.0 - gate_j)
                dL_dthreshold[i] = -steepness * w * (dL_dgate_i[:, None] * surrogate).sum(axis=0)
                term = dL_dgate_i[:, None] * (gate_j - gate[:, None])
                dL_dalpha[i] = w * term.sum(axis=0)
            current_g = next_g

        alpha -= lr * np.array(dL_dalpha)
        threshold -= lr * np.array(dL_dthreshold)
        leaf_logits -= lr * np.array(dL_dleaf_logits)

    return {"alpha": alpha, "threshold": threshold, "leaf_logits": leaf_logits, "steepness": steepness, "depth": depth}


def predict(X: np.ndarray, params: dict) -> np.ndarray:
    depth = params["depth"]
    steepness = params["steepness"]
    n_samples = X.shape[0]
    current_level_probs = [np.ones(n_samples)]
    node_idx = 0
    for _level in range(depth):
        next_level_probs = []
        for parent_prob_arr in current_level_probs:
            w = softmax(params["alpha"][node_idx])
            gate_j = 1.0 / (1.0 + np.exp(-steepness * (X - params["threshold"][node_idx])))
            gate = gate_j @ w
            next_level_probs.append(parent_prob_arr * (1.0 - gate))
            next_level_probs.append(parent_prob_arr * gate)
            node_idx += 1
        current_level_probs = next_level_probs
    leaf_probs = current_level_probs
    leafdist = [softmax(params["leaf_logits"][l]) for l in range(len(leaf_probs))]
    y_hat = sum(np.outer(leaf_probs[l], leafdist[l]) for l in range(len(leaf_probs)))
    return y_hat.argmax(axis=1)


def main():
    for dataset_name in ["iris", "wine", "breast_cancer"]:
        X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name)
        n_classes = int(max(y_train.max(), y_test.max()) + 1)
        y_train_oh = one_hot_encode(y_train, n_classes)
        for depth in [1, 2, 3]:
            params = train_depthN(X_train, y_train_oh, depth=depth)
            train_acc = (predict(X_train, params) == y_train).mean()
            test_acc = (predict(X_test, params) == y_test).mean()
            print(f"{dataset_name:<15} depth={depth}  train={train_acc:.4f}  test={test_acc:.4f}")


if __name__ == "__main__":
    main()
