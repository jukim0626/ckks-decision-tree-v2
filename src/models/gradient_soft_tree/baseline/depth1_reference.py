"""depth=1 gradient soft tree의 CKKS-매칭 plaintext 기준 구현.

`soft_tree_gd.py`(torch, cross-entropy+Adam, autograd)와 다르다 - 이 파일은 CKKS로 그대로
옮길 걸 염두에 두고 **CKKS에서 실제로 계산 가능한 연산만** 손으로 미분해서 짰다:
- loss: cross-entropy 대신 MSE (log를 다항식 근사할 필요가 없어서 CKKS에 훨씬 안전)
- sigmoid 미분: 실제 도함수 sigmoid'(z)=sigmoid(z)(1-sigmoid(z)) 대신, **다항식 근사로 이미
  계산해둔 gate 값 자체로 근사한 surrogate `gate*(1-gate)`** 사용 (다항식을 또 미분해서 새
  근사를 만들 필요 없이, forward에서 만든 값을 재사용 - 근사의 근사를 새로 만드는 것보다
  일관적이고 구현 비용도 적음. 근사 활성함수를 쓰는 encrypted/quantized NN 학습에서 흔한
  surrogate gradient 트릭과 같은 발상)
- optimizer: Adam 아니고 plain SGD (CKKS판에서 Adam의 rsqrt(2차 모멘트) 안전 범위 설계는
  추가 작업이라 1단계에서는 뺐다 - 이 계산 자체가 맞는지부터 검증하는 게 우선)

CKKS 버전(`depth1_ckks.py`)은 이 파일의 forward/backward 수식을 한 줄씩 그대로 암호문
연산으로 옮긴 것 - 두 파일의 각 변수명이 대응되게 맞춰뒀다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import load_scaled_dataset_subset, one_hot_encode  # noqa: E402


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def softmax_backward(dist: np.ndarray, dL_ddist: np.ndarray) -> np.ndarray:
    """d(softmax)/d(logit) 야코비안을 적용: dL/dlogit_k = dist_k*(dL_ddist_k - sum_c dL_ddist_c*dist_c)."""
    s = float(np.dot(dL_ddist, dist))
    return dist * (dL_ddist - s)


def train_depth1(
    X: np.ndarray,
    y_onehot: np.ndarray,
    steepness: float = 8.0,
    lr: float = 0.3,
    epochs: int = 300,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    n_classes = y_onehot.shape[1]

    alpha = rng.normal(0, 0.1, size=n_features)
    threshold = rng.normal(0, 0.1, size=n_features)
    leaf_L = rng.normal(0, 0.1, size=n_classes)
    leaf_R = rng.normal(0, 0.1, size=n_classes)

    for _ in range(epochs):
        w = softmax(alpha)  # (n_features,)
        gate_j = 1.0 / (1.0 + np.exp(-steepness * (X - threshold)))  # (n, n_features) - 실제로는 다항식 근사가 들어갈 자리
        gate = gate_j @ w  # (n,) - blended "오른쪽으로 갈 확률"
        left_prob, right_prob = 1.0 - gate, gate

        leafdist_L, leafdist_R = softmax(leaf_L), softmax(leaf_R)
        y_hat = np.outer(left_prob, leafdist_L) + np.outer(right_prob, leafdist_R)  # (n, n_classes)

        dL_dyhat = (2.0 / n_samples) * (y_hat - y_onehot)  # (n, n_classes), MSE gradient

        dL_dleafdist_L = dL_dyhat.T @ left_prob  # (n_classes,)
        dL_dleafdist_R = dL_dyhat.T @ right_prob

        dL_dleaf_L = softmax_backward(leafdist_L, dL_dleafdist_L)
        dL_dleaf_R = softmax_backward(leafdist_R, dL_dleafdist_R)

        dL_dgate = dL_dyhat @ (leafdist_R - leafdist_L)  # (n,)

        surrogate = gate_j * (1.0 - gate_j)  # (n, n_features) - sigmoid'(z)의 surrogate
        dL_dt = -steepness * w * (dL_dgate[:, None] * surrogate).sum(axis=0)  # (n_features,)

        term = dL_dgate[:, None] * (gate_j - gate[:, None])  # (n, n_features)
        dL_dalpha = w * term.sum(axis=0)  # (n_features,)

        alpha -= lr * dL_dalpha
        threshold -= lr * dL_dt
        leaf_L -= lr * dL_dleaf_L
        leaf_R -= lr * dL_dleaf_R

    return {"alpha": alpha, "threshold": threshold, "leaf_L": leaf_L, "leaf_R": leaf_R, "steepness": steepness}


def predict(X: np.ndarray, params: dict) -> np.ndarray:
    w = softmax(params["alpha"])
    gate_j = 1.0 / (1.0 + np.exp(-params["steepness"] * (X - params["threshold"])))
    gate = gate_j @ w
    left_prob, right_prob = 1.0 - gate, gate
    leafdist_L, leafdist_R = softmax(params["leaf_L"]), softmax(params["leaf_R"])
    y_hat = np.outer(left_prob, leafdist_L) + np.outer(right_prob, leafdist_R)
    return y_hat.argmax(axis=1)


def main():
    for dataset_name in ["iris", "wine", "breast_cancer"]:
        X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name)
        n_classes = int(max(y_train.max(), y_test.max()) + 1)
        y_train_oh = one_hot_encode(y_train, n_classes)

        params = train_depth1(X_train, y_train_oh)
        train_acc = (predict(X_train, params) == y_train).mean()
        test_acc = (predict(X_test, params) == y_test).mean()
        print(f"{dataset_name:<15} depth=1  train={train_acc:.4f}  test={test_acc:.4f}")


if __name__ == "__main__":
    main()
