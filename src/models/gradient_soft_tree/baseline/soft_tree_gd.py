"""MGI(Gini) 계산 없는 진짜 end-to-end soft decision tree 프로토타입 (plaintext, CKKS 아님).

`closed_form_mgi`(현재 production 계보)는 각 노드에서 여전히 "Gini를 최소화하는 분기를
찾는다"는 hard-tree의 논리를 그대로 쓰고(각 feature의 MGI 점수를 계산 -> softmax로 blend),
sigmoid는 그 위에 씌운 코팅일 뿐이다 (2026-08-16/17 EXPERIMENT_LOG.md 실험에서 이 구조
안에서 "MGI 목적함수 + vanilla SGD"로 노드 하나만 갱신해본 적은 있지만 실패했음 - mean
0.878, 학습률에 민감. 원인: 스텝 크기가 목적함수의 국소 분산을 반영 못 함 = Adam이
2차 모멘트로 스텝을 정규화하는 이유와 동일).

이 스크립트는 그 실패를 정면으로 고친 버전이다:
- MGI/Gini 계산을 아예 안 함 (softmax-blend-of-Gini-scores 없음, closed-form threshold 없음)
- 목적함수를 Gini 대리 지표가 아니라 최종 leaf의 실제 cross-entropy loss로 둠
- vanilla SGD 대신 Adam(2차 모멘트 정규화)으로 갱신
- 노드 하나가 아니라 트리 전체(모든 internal node + 모든 leaf)를 조인트로 학습

구조는 hierarchical mixture of experts(Jordan & Jacobs 1994)와 동일: 각 internal node가
"오른쪽으로 갈 확률"(=blended sigmoid gate, feature별 sigmoid를 학습 가능한 softmax
attention(alpha)으로 가중합)을 내고, root->leaf 경로 확률의 곱으로 각 leaf의 도달 확률을
구한 뒤, leaf별 학습 가능한 class 분포를 그 확률로 섞어 최종 예측을 만든다. 전부
미분가능이라 cross-entropy loss를 놓고 Adam으로 바로 backprop한다.

**"숫자형 데이터 특화" 설계**: node의 split을 범용 선형결합(w·x+b, 카테고리형에 흔히 쓰는
형태)이 아니라 지금 프로젝트처럼 feature 하나하나를 그 feature의 threshold와 비교하는
axis-aligned 구조로 유지했다 (`sigmoid(steepness*(x_j - threshold_j))`를 feature별로 계산해
softmax attention(alpha)으로 가중합) - 순서/크기가 의미 있는 수치형 feature의 특성을 그대로
활용하는 결정, 카테고리형이면 자연스러운 "이 값보다 큰가"라는 비교 자체가 성립 안 하므로
w·x+b 쪽이 더 맞겠지만 이 프로젝트 데이터(iris/wine/breast_cancer)는 전부 수치형이라 이
구조가 더 적합하다는 것이 교수님 코멘트와 일치.

**steepness annealing**: threshold가 초기엔 random이라 데이터와 거리가 멀면 sigmoid가
포화되어 gradient가 거의 0이 되는 문제가 있다 (특히 steepness가 크면 심함). 학습 초반엔
완만한 sigmoid(steepness 작게)로 gradient가 잘 흐르게 하고, 후반으로 갈수록 steepness를
선형으로 키워 hard-threshold에 가깝게 sharpen한다 (differentiable relaxation 학습의 표준
기법 - Gumbel-softmax의 temperature annealing과 같은 원리).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import load_scaled_dataset_subset  # noqa: E402


class GradientSoftTree(nn.Module):
    def __init__(self, n_features: int, n_classes: int, depth: int):
        super().__init__()
        self.depth = depth
        self.n_internal = (1 << depth) - 1
        self.n_leaves = 1 << depth
        # alpha: node별 feature attention logit (softmax하면 "이 노드가 어떤 feature를
        # 보는가"). threshold: node x feature별 학습 가능한 분기 기준값.
        # 0으로 초기화하면 seed와 무관하게 완전히 같은 지점에서 출발해서(대칭 초기화라
        # gradient도 결정론적) "여러 seed로 안정성 확인"이 사실상 무의미해진다 - 작은
        # random noise로 실제 seed별 다른 시작점을 만든다.
        self.alpha = nn.Parameter(torch.randn(self.n_internal, n_features) * 0.1)
        self.threshold = nn.Parameter(torch.randn(self.n_internal, n_features) * 0.1)
        self.leaf_logits = nn.Parameter(torch.randn(self.n_leaves, n_classes) * 0.1)

    def forward(self, x: torch.Tensor, steepness: float) -> torch.Tensor:
        level_probs = [torch.ones(x.shape[0], dtype=x.dtype, device=x.device)]
        node_id = 0
        for _ in range(self.depth):
            next_level = []
            for parent_prob in level_probs:
                w = torch.softmax(self.alpha[node_id], dim=0)
                per_feature_gate = torch.sigmoid(steepness * (x - self.threshold[node_id]))
                gate = (per_feature_gate * w).sum(dim=1)
                next_level.append(parent_prob * (1.0 - gate))
                next_level.append(parent_prob * gate)
                node_id += 1
            level_probs = next_level
        leaf_reach_prob = torch.stack(level_probs, dim=1)  # (N, n_leaves)
        leaf_dist = torch.softmax(self.leaf_logits, dim=1)  # (n_leaves, n_classes)
        return leaf_reach_prob @ leaf_dist  # (N, n_classes), 이미 확률(합=1)


def train_gradient_soft_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    depth: int,
    epochs: int = 800,
    lr: float = 0.05,
    steepness_start: float = 1.0,
    steepness_end: float = 8.0,
    seed: int = 0,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    n_features = X_train.shape[1]
    n_classes = int(max(y_train.max(), y_test.max()) + 1)

    model = GradientSoftTree(n_features, n_classes, depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    for epoch in range(epochs):
        steepness = steepness_start + (steepness_end - steepness_start) * (epoch / max(epochs - 1, 1))
        optimizer.zero_grad()
        y_hat = model(X_train_t, steepness)
        loss = -torch.log(y_hat[torch.arange(len(y_train_t)), y_train_t] + 1e-9).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_pred = model(X_train_t, steepness_end).argmax(dim=1)
        test_pred = model(X_test_t, steepness_end).argmax(dim=1)
        train_acc = (train_pred == y_train_t).float().mean().item()
        test_acc = (test_pred == y_test_t).float().mean().item()
    return train_acc, test_acc


def main():
    datasets = ["iris", "wine", "breast_cancer"]
    depths = [1, 2, 3]
    seeds = [0, 1, 2, 3, 4]
    print(f"{'dataset':<15}{'depth':<7}{'sklearn hard':<15}{'grad-soft-tree test acc (mean±std over 5 seeds)':<48}")
    for dataset_name in datasets:
        X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name)
        for depth in depths:
            clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
            clf.fit(X_train, y_train)
            hard_acc = clf.score(X_test, y_test)

            test_accs = [
                train_gradient_soft_tree(X_train, y_train, X_test, y_test, depth=depth, seed=seed)[1]
                for seed in seeds
            ]
            mean_acc = float(np.mean(test_accs))
            std_acc = float(np.std(test_accs))
            print(f"{dataset_name:<15}{depth:<7}{hard_acc:<15.4f}{mean_acc:.4f} ± {std_acc:.4f}  {test_accs}")


if __name__ == "__main__":
    main()
