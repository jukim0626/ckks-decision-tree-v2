"""depthN_reference.py(axis-aligned attention-blend gate)의 gate 파라미터화만 바꾼
oblique(선형결합) 버전. 나머지(leaf softmax, level-by-level backward 재귀 구조,
CKKS-매칭 스타일)는 전부 동일하게 유지한다 - "gate가 어떻게 만들어지는지"만 다르다.

**기존(axis-aligned attention-blend) gate**:
  gate_j = sigmoid(steepness * (x_j - threshold_j))   # feature마다 따로 판단
  gate   = sum_j  softmax(alpha)_j * gate_j            # 판단들을 나중에 섞음
  -> 파라미터: alpha(n_features), threshold(n_features) 두 벌, softmax 필요

**oblique(선형결합) gate**:
  z    = w . x + b                                     # feature '값'을 먼저 섞음
  gate = sigmoid(steepness * z)                        # 섞은 다음 딱 한 번 판단
  -> 파라미터: w(n_features), b(스칼라) 한 벌, softmax 불필요

이 두 gate는 수학적으로 다르다(sigmoid는 비선형이라 "sigmoid의 합" != "합의 sigmoid") -
oblique gate는 노드 하나로 기울어진(oblique) 결정 경계를 그릴 수 있어서, 축에 평행한
gate 여러 개를 섞어야만 근사할 수 있던 경계를 더 얕은 depth로 표현할 수 있다는 게 가설.

**backward 유도** (dL_dgate_i는 기존과 완전히 동일 - depthN_reference.py의 레벨별 재귀
공식 그대로 재사용, 여기서 바뀌는 건 "dL_dgate_i를 받아서 파라미터 gradient로 바꾸는 마지막
한 단계"뿐):
  gate = sigmoid(steepness * z),  z = w.x + b
  dgate/dz = steepness * gate * (1 - gate)
  dL/dz    = dL_dgate_i * dgate/dz                      # (n_samples,)
  dL/dw    = X.T @ (dL/dz)                              # (n_features,) - 기존의 softmax
                                                         #   backward(alpha 쪽 gradient)가
                                                         #   통째로 사라지고 이 한 줄로 대체됨
  dL/db    = sum(dL/dz)                                 # 기존엔 없던 새 파라미터(threshold의
                                                         #   역할을 겸함)

**2026-09-07 안정성 수정 3종** (iris depth=2 CKKS 30-epoch 실험에서 root 노드가 포화되며
plaintext 대비 max_err=0.19까지 벌어진 사건 이후 추가 - `z = w.x+b`가 feature 개수만큼
항을 더하는 구조라 feature가 많을수록(또는 학습이 오래 진행될수록) `z`가 CKKS sigmoid
다항식의 안전구간([-2,2])을 벗어나기 쉬워진다는 게 원인으로 추정됨):

1. **초기화 스케일링**: `w`의 초기 표준편차를 고정값(0.1) 대신 `0.1/sqrt(n_features)`로.
   Var(z) = n_features * Var(w_j) * Var(x_j)이므로 Var(w_j) ∝ 1/n_features로 잡아야
   feature 개수와 무관하게 z의 분산이 일정해짐(신경망의 Xavier/Glorot 초기화와 같은 원리).
2. **gate 쪽 lr도 같은 비율로 스케일링**: `lr_gate = lr / sqrt(n_features)`. 초기화만
   고치면 시작점은 안전해도 학습이 진행되며(feature 많을수록 항이 많이 쌓여) 다시 벗어날
   수 있어서, w/b 업데이트 폭 자체도 줄인다. leaf_logits는 feature 개수와 무관한 파라미터라
   원래 lr을 그대로 쓴다.
3. **L2 weight decay**: 매 업데이트에 `dL_dw += weight_decay*w`, `dL_db += weight_decay*b`를
   더해서 w,b가 무한정 커지는 걸 계속 눌러준다(CKKS 노이즈로 인한 편향된 드리프트까지
   완전히 막지는 못해도 원점 쪽으로 당기는 자기교정 힘 역할). leaf_logits는 decay 안 함.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from client_assisted.dataset import load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from experiments.gradient_soft_tree.depth1_reference import softmax, softmax_backward  # noqa: E402


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def train_depthN_oblique(
    X: np.ndarray,
    y_onehot: np.ndarray,
    depth: int,
    steepness: float = 8.0,
    lr: float = 0.3,
    epochs: int = 300,
    seed: int = 0,
    weight_decay: float = 0.01,
) -> dict:
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    n_classes = y_onehot.shape[1]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth
    lr_gate = lr / np.sqrt(n_features)  # 안정성 수정 #2 - gate(w,b) 전용 lr

    w = rng.normal(0, 0.1 / np.sqrt(n_features), size=(n_internal, n_features))  # 안정성 수정 #1
    b = rng.normal(0, 0.1, size=(n_internal,))
    leaf_logits = rng.normal(0, 0.1, size=(n_leaves, n_classes))

    for _ in range(epochs):
        # ---- forward ----
        node_gate = [None] * n_internal
        node_parent_prob = [None] * n_internal

        current_level_probs = [np.ones(n_samples)]
        node_idx = 0
        for _level in range(depth):
            next_level_probs = []
            for parent_prob_arr in current_level_probs:
                i = node_idx
                z = X @ w[i] + b[i]
                gate = sigmoid(steepness * z)
                node_gate[i] = gate
                node_parent_prob[i] = parent_prob_arr
                next_level_probs.append(parent_prob_arr * (1.0 - gate))
                next_level_probs.append(parent_prob_arr * gate)
                node_idx += 1
            current_level_probs = next_level_probs
        leaf_probs = current_level_probs

        leafdist = [softmax(leaf_logits[l]) for l in range(n_leaves)]
        y_hat = sum(np.outer(leaf_probs[l], leafdist[l]) for l in range(n_leaves))
        dL_dyhat = (2.0 / n_samples) * (y_hat - y_onehot)

        # ---- leaf gradients + g(=dL/dp_leaf) : depthN_reference.py와 완전히 동일 ----
        current_g = []
        dL_dleaf_logits = [None] * n_leaves
        for l in range(n_leaves):
            dL_ddist_l = dL_dyhat.T @ leaf_probs[l]
            dL_dleaf_logits[l] = softmax_backward(leafdist[l], dL_ddist_l)
            current_g.append(dL_dyhat @ leafdist[l])

        # ---- 레벨을 거슬러 올라가며 노드 gradient 계산 (gate 부분만 다름) ----
        dL_dw = [None] * n_internal
        dL_db = [None] * n_internal
        for level in reversed(range(depth)):
            start = (1 << level) - 1
            count = 1 << level
            next_g = [None] * count
            for idx in range(count):
                i = start + idx
                g_left, g_right = current_g[2 * idx], current_g[2 * idx + 1]
                gate = node_gate[i]
                p_i = node_parent_prob[i]

                dL_dgate_i = p_i * (g_right - g_left)  # depthN_reference.py와 동일
                next_g[idx] = g_left * (1.0 - gate) + g_right * gate

                dL_dz = steepness * dL_dgate_i * gate * (1.0 - gate)  # (n_samples,)
                dL_dw[i] = X.T @ dL_dz + weight_decay * w[i]  # 안정성 수정 #3
                dL_db[i] = dL_dz.sum() + weight_decay * b[i]
            current_g = next_g

        w -= lr_gate * np.array(dL_dw)
        b -= lr_gate * np.array(dL_db)
        leaf_logits -= lr * np.array(dL_dleaf_logits)

    return {"w": w, "b": b, "leaf_logits": leaf_logits, "steepness": steepness, "depth": depth}


def predict(X: np.ndarray, params: dict) -> np.ndarray:
    depth = params["depth"]
    steepness = params["steepness"]
    n_samples = X.shape[0]
    current_level_probs = [np.ones(n_samples)]
    node_idx = 0
    for _level in range(depth):
        next_level_probs = []
        for parent_prob_arr in current_level_probs:
            z = X @ params["w"][node_idx] + params["b"][node_idx]
            gate = sigmoid(steepness * z)
            next_level_probs.append(parent_prob_arr * (1.0 - gate))
            next_level_probs.append(parent_prob_arr * gate)
            node_idx += 1
        current_level_probs = next_level_probs
    leaf_probs = current_level_probs
    leafdist = [softmax(params["leaf_logits"][l]) for l in range(len(leaf_probs))]
    y_hat = sum(np.outer(leaf_probs[l], leafdist[l]) for l in range(len(leaf_probs)))
    return y_hat.argmax(axis=1)


def _grad_check():
    """finite-difference로 dL_dw/dL_db 공식이 맞는지 확인 (실제 학습 전 1회성 검증)."""
    rng = np.random.default_rng(0)
    n_samples, n_features, n_classes, depth = 12, 4, 3, 2
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, n_classes, size=n_samples)
    y_onehot = np.eye(n_classes)[y]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth
    steepness = 8.0

    w = rng.normal(0, 0.5, size=(n_internal, n_features))
    b = rng.normal(0, 0.5, size=(n_internal,))
    leaf_logits = rng.normal(0, 0.5, size=(n_leaves, n_classes))
    weight_decay = 0.01

    def loss_fn(w, b, leaf_logits):
        current_level_probs = [np.ones(n_samples)]
        node_idx = 0
        for _level in range(depth):
            next_level_probs = []
            for parent_prob_arr in current_level_probs:
                z = X @ w[node_idx] + b[node_idx]
                gate = sigmoid(steepness * z)
                next_level_probs.append(parent_prob_arr * (1.0 - gate))
                next_level_probs.append(parent_prob_arr * gate)
                node_idx += 1
            current_level_probs = next_level_probs
        leaf_probs = current_level_probs
        leafdist = [softmax(leaf_logits[l]) for l in range(n_leaves)]
        y_hat = sum(np.outer(leaf_probs[l], leafdist[l]) for l in range(n_leaves))
        mse = ((y_hat - y_onehot) ** 2).sum() / n_samples
        l2 = 0.5 * weight_decay * ((w ** 2).sum() + (b ** 2).sum())  # 안정성 수정 #3과 대응
        return mse + l2

    # ---- analytic gradient (train_depthN_oblique 본문과 동일한 계산을 1 epoch만 재현) ----
    node_gate = [None] * n_internal
    node_parent_prob = [None] * n_internal
    current_level_probs = [np.ones(n_samples)]
    node_idx = 0
    for _level in range(depth):
        next_level_probs = []
        for parent_prob_arr in current_level_probs:
            i = node_idx
            z = X @ w[i] + b[i]
            gate = sigmoid(steepness * z)
            node_gate[i] = gate
            node_parent_prob[i] = parent_prob_arr
            next_level_probs.append(parent_prob_arr * (1.0 - gate))
            next_level_probs.append(parent_prob_arr * gate)
            node_idx += 1
        current_level_probs = next_level_probs
    leaf_probs = current_level_probs
    leafdist = [softmax(leaf_logits[l]) for l in range(n_leaves)]
    y_hat = sum(np.outer(leaf_probs[l], leafdist[l]) for l in range(n_leaves))
    dL_dyhat = (2.0 / n_samples) * (y_hat - y_onehot)

    current_g = []
    for l in range(n_leaves):
        current_g.append(dL_dyhat @ leafdist[l])

    dL_dw = [None] * n_internal
    dL_db = [None] * n_internal
    for level in reversed(range(depth)):
        start = (1 << level) - 1
        count = 1 << level
        next_g = [None] * count
        for idx in range(count):
            i = start + idx
            g_left, g_right = current_g[2 * idx], current_g[2 * idx + 1]
            gate = node_gate[i]
            p_i = node_parent_prob[i]
            dL_dgate_i = p_i * (g_right - g_left)
            next_g[idx] = g_left * (1.0 - gate) + g_right * gate
            dL_dz = steepness * dL_dgate_i * gate * (1.0 - gate)
            dL_dw[i] = X.T @ dL_dz + weight_decay * w[i]
            dL_db[i] = dL_dz.sum() + weight_decay * b[i]
        current_g = next_g
    dL_dw = np.array(dL_dw)
    dL_db = np.array(dL_db)

    eps = 1e-5
    max_err_w = 0.0
    for i in range(n_internal):
        for j in range(n_features):
            w_plus = w.copy(); w_plus[i, j] += eps
            w_minus = w.copy(); w_minus[i, j] -= eps
            numeric = (loss_fn(w_plus, b, leaf_logits) - loss_fn(w_minus, b, leaf_logits)) / (2 * eps)
            max_err_w = max(max_err_w, abs(numeric - dL_dw[i, j]))

    max_err_b = 0.0
    for i in range(n_internal):
        b_plus = b.copy(); b_plus[i] += eps
        b_minus = b.copy(); b_minus[i] -= eps
        numeric = (loss_fn(w, b_plus, leaf_logits) - loss_fn(w, b_minus, leaf_logits)) / (2 * eps)
        max_err_b = max(max_err_b, abs(numeric - dL_db[i]))

    print(f"[grad_check] max abs error  dL_dw={max_err_w:.3e}  dL_db={max_err_b:.3e}  (should be ~1e-6~1e-8)")


def main():
    _grad_check()
    print()
    for dataset_name in ["iris", "wine", "breast_cancer"]:
        X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name)
        n_classes = int(max(y_train.max(), y_test.max()) + 1)
        y_train_oh = one_hot_encode(y_train, n_classes)
        for depth in [1, 2, 3]:
            params = train_depthN_oblique(X_train, y_train_oh, depth=depth)
            train_acc = (predict(X_train, params) == y_train).mean()
            test_acc = (predict(X_test, params) == y_test).mean()
            n_internal = (1 << depth) - 1
            print(f"{dataset_name:<15} depth={depth} (nodes={n_internal})  train={train_acc:.4f}  test={test_acc:.4f}")


if __name__ == "__main__":
    main()
