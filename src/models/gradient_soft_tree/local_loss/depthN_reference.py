"""ReBoot(arXiv 2506.19693)의 local-loss block 아이디어를 gradient_soft_tree에 적용한
plaintext 프로토타입. baseline `depthN_reference.py`(단일 최종 leaf loss로 전체 트리를
end-to-end backprop)는 절대 안 건드림 - 이건 완전히 다른 학습 알고리즘을 시도하는 새 파일.

**핵심 아이디어**: 기존엔 트리 전체를 통과한 최종 leaf에서만 loss를 계산해서 root까지
전부 역전파했다(backward 깊이가 tree depth에 비례). ReBoot처럼 **레벨마다 자기만의 local
classifier를 붙여서 그 레벨에서 바로 loss를 계산**하면, 각 레벨의 gate 파라미터
(alpha/threshold)는 **그 레벨 자신의 local loss에서만** gradient를 받는다 - 더 깊은
레벨의 loss가 얕은 레벨의 파라미터까지 역전파해서 내려오지 않는다(명시적으로 끊는 게
아니라, 애초에 그 경로로 gradient를 계산하지 않는 구조 - manual backprop이라 자연스럽게
그렇게 됨).

**forward는 baseline과 100% 동일**: 각 노드의 gate는 여전히 feature별 sigmoid를
softmax attention(alpha)로 blend하는 것(baseline이 이미 CKKS에서 안정적으로 검증된
형태 - 오블리크처럼 여러 feature를 날것으로 더하는 구조가 아니라서 z가 커지는 문제
자체가 없음). depth가 깊어져도 개별 gate의 z 범위는 baseline과 완전히 동일 - 이 실험은
"학습 알고리즘"(loss를 어디서 계산하고 gradient를 얼마나 멀리 보낼지)만 바꾼다.

**local classifier 설계**: level ℓ(0-indexed, 0..depth-1)을 통과하고 나면 2^(ℓ+1)개의
"레벨 ℓ까지의 가상 leaf"에 도달할 확률이 생긴다. 이 가상 leaf마다 자기만의 분류기
파라미터(`local_logits[ℓ]`, shape (2^(ℓ+1), n_classes))를 두고, baseline의 진짜 leaf와
완전히 같은 방식(softmax 후 reach-prob로 가중합)으로 그 레벨의 예측(y_hat_level)과
loss를 계산한다. depth-1번째(가장 깊은) local classifier가 baseline의 leaf_logits와
동일한 역할을 한다.

**주의(2026-09-09 제기된 리스크)**: 이건 본질적으로 레벨별 독립 최적화라 greedy에
가까운 성격이 있다 - [[project_ckks_decision_tree]]에서 "MGI+greedy보다 gradient+joint가
정확도를 크게 높였다"는 기존 발견과 충돌할 위험이 있어서, CKKS로 옮기기 전에 반드시
이 plaintext 비교로 정확도 손실 여부부터 확인해야 한다."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from models.gradient_soft_tree.depth1_reference import softmax, softmax_backward  # noqa: E402
from models.gradient_soft_tree.depthN_reference import train_depthN, predict as predict_baseline  # noqa: E402


def train_depthN_local_loss(
    X: np.ndarray,
    y_onehot: np.ndarray,
    depth: int,
    steepness: float = 8.0,
    lr: float = 0.3,
    epochs: int = 300,
    seed: int = 0,
    level_weights: list[float] | None = None,
) -> dict:
    """level_weights: 레벨별 local loss 가중치(기본은 전부 1.0 - 논문처럼 레벨마다 동등하게
    취급). 더 깊은 레벨(최종 예측에 가까움)에 더 큰 가중치를 주고 싶으면 조정 가능."""
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    n_classes = y_onehot.shape[1]
    n_internal = (1 << depth) - 1
    if level_weights is None:
        level_weights = [1.0] * depth

    alpha = rng.normal(0, 0.1, size=(n_internal, n_features))
    threshold = rng.normal(0, 0.1, size=(n_internal, n_features))
    local_logits = [rng.normal(0, 0.1, size=(1 << (lvl + 1), n_classes)) for lvl in range(depth)]

    for _ in range(epochs):
        dL_dalpha = [None] * n_internal
        dL_dthreshold = [None] * n_internal
        dL_dlocal_logits = [None] * depth

        current_level_probs = [np.ones(n_samples)]
        node_idx = 0
        for level in range(depth):
            start = (1 << level) - 1
            count = 1 << level
            next_level_probs = []
            node_gate = [None] * count
            node_gate_j = [None] * count
            node_w = [None] * count
            node_parent_prob = [None] * count
            for idx, parent_prob_arr in enumerate(current_level_probs):
                i = node_idx
                w = softmax(alpha[i])
                gate_j = 1.0 / (1.0 + np.exp(-steepness * (X - threshold[i])))
                gate = gate_j @ w
                node_gate[idx], node_gate_j[idx], node_w[idx] = gate, gate_j, w
                node_parent_prob[idx] = parent_prob_arr
                next_level_probs.append(parent_prob_arr * (1.0 - gate))
                next_level_probs.append(parent_prob_arr * gate)
                node_idx += 1
            current_level_probs = next_level_probs  # 이 레벨까지의 "가상 leaf" reach prob (길이 2^(level+1))

            # ---- 이 레벨의 local loss(그 자체로 완결된 얕은 트리 예측) ----
            local_dist = [softmax(local_logits[level][k]) for k in range(len(current_level_probs))]
            y_hat_level = sum(
                np.outer(current_level_probs[k], local_dist[k]) for k in range(len(current_level_probs))
            )
            dL_dyhat_level = level_weights[level] * (2.0 / n_samples) * (y_hat_level - y_onehot)

            g_this_level = [None] * len(current_level_probs)
            local_logits_grad = np.zeros_like(local_logits[level])
            for k in range(len(current_level_probs)):
                dL_ddist_k = dL_dyhat_level.T @ current_level_probs[k]
                local_logits_grad[k] = softmax_backward(local_dist[k], dL_ddist_k)
                g_this_level[k] = dL_dyhat_level @ local_dist[k]
            dL_dlocal_logits[level] = local_logits_grad

            # ---- 이 레벨의 gate 파라미터(alpha/threshold)는 g_this_level에서만 gradient를
            # 받는다 - 더 깊은 레벨(level+1, level+2, ...)의 local loss는 여기 관여 안 함.
            for idx in range(count):
                i = start + idx
                g_left, g_right = g_this_level[2 * idx], g_this_level[2 * idx + 1]
                gate, gate_j, w = node_gate[idx], node_gate_j[idx], node_w[idx]
                p_i = node_parent_prob[idx]

                dL_dgate_i = p_i * (g_right - g_left)
                surrogate = gate_j * (1.0 - gate_j)
                dL_dthreshold[i] = -steepness * w * (dL_dgate_i[:, None] * surrogate).sum(axis=0)
                term = dL_dgate_i[:, None] * (gate_j - gate[:, None])
                dL_dalpha[i] = w * term.sum(axis=0)

        alpha -= lr * np.array(dL_dalpha)
        threshold -= lr * np.array(dL_dthreshold)
        for level in range(depth):
            local_logits[level] -= lr * dL_dlocal_logits[level]

    return {
        "alpha": alpha, "threshold": threshold, "leaf_logits": local_logits[-1],
        "steepness": steepness, "depth": depth,
    }


def predict(X: np.ndarray, params: dict) -> np.ndarray:
    """가장 깊은 레벨의 local classifier(=leaf_logits)를 최종 예측으로 사용 - baseline
    predict()와 동일한 형태(params dict 모양이 같아서 predict_baseline 그대로 재사용 가능)."""
    return predict_baseline(X, params)


def main():
    for dataset_name in ["iris", "wine", "breast_cancer"]:
        X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name)
        n_classes = int(max(y_train.max(), y_test.max()) + 1)
        y_train_oh = one_hot_encode(y_train, n_classes)
        for depth in [1, 2, 3]:
            baseline_params = train_depthN(X_train, y_train_oh, depth=depth)
            baseline_train_acc = (predict_baseline(X_train, baseline_params) == y_train).mean()
            baseline_test_acc = (predict_baseline(X_test, baseline_params) == y_test).mean()

            local_params = train_depthN_local_loss(X_train, y_train_oh, depth=depth)
            local_train_acc = (predict(X_train, local_params) == y_train).mean()
            local_test_acc = (predict(X_test, local_params) == y_test).mean()

            print(
                f"{dataset_name:<15} depth={depth}  "
                f"baseline(joint) train={baseline_train_acc:.4f} test={baseline_test_acc:.4f}  |  "
                f"local-loss train={local_train_acc:.4f} test={local_test_acc:.4f}"
            )


if __name__ == "__main__":
    main()
