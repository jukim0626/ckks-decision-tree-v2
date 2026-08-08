"""soft-MGI(softmax blending, sign_bootstrap/argmin 없음)의 plaintext 버전 (depth>=1 지원).

`soft_mgi.py`(CKKS 버전, depth=1 stump만 지원)를 암호화 없이 numpy로 재현해서, CKKS로
포팅하기 전에 (a) depth>=2로 재귀 확장이 실제로 잘 되는지, (b) beta/depth/dataset
조합에서 어떤 게 잘 되는지를 빠르게 iterate하기 위한 실험용 모듈. 수식은
`fully_encrypted_mgi_stump.encrypted_mgi_from_counts`(division-free MGI,
|S|^2-sum_c|S_c|^2)와 `soft_mgi.soft_mgi_weights`(softmax weight)를 그대로 따른다.

**노드마다 candidate 풀은 동일**(client_assisted.training과 같은 패턴 - 매 노드 같은
public grid를 node_weight만 바꿔서 재평가). **score_normalizer는 모든 노드에서
n_samples**2로 고정**(root 기준 공개 상한) - node_weight<=1이라 실제 weighted count는
항상 unweighted count 이하이므로, 깊은 노드에서는 느슨하지만(tight하지 않지만) 여전히
유효한 상한이다. 이렇게 하면 normalizer가 데이터에 의존하지 않고 매 노드 동일한 공개
상수로 남아서, 나중에 CKKS로 포팅할 때도 그대로 쓸 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from client_assisted.candidates import PublicSplitCandidate

STEEPNESS = 8.0


def sigmoid_gate(x_col: np.ndarray, threshold: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-STEEPNESS * (x_col - threshold)))


def mgi(counts: np.ndarray) -> float:
    total = counts.sum()
    return float(total**2 - (counts**2).sum())


@dataclass
class SoftMgiSplitNode:
    weights: np.ndarray  # softmax weight per candidate (len == len(candidates))
    scores: np.ndarray  # MGI score per candidate (진단용)
    left: "SoftMgiNode"
    right: "SoftMgiNode"


@dataclass
class SoftMgiLeafNode:
    counts: np.ndarray  # class별 weighted count


SoftMgiNode = SoftMgiSplitNode | SoftMgiLeafNode


@dataclass
class SoftMgiHardBaselineSplitNode:
    """비교용: softmax 대신 argmin(1등만 weight=1)으로 노드를 만드는 hard 버전."""

    winner_idx: int
    scores: np.ndarray
    left: "SoftMgiHardNode"
    right: "SoftMgiHardNode"


SoftMgiHardNode = SoftMgiHardBaselineSplitNode | SoftMgiLeafNode


def _evaluate_candidates(X, y_onehot, candidates, node_weight):
    gates = []
    left_counts_list = []
    right_counts_list = []
    scores = np.zeros(len(candidates))
    for idx, c in enumerate(candidates):
        rp = sigmoid_gate(X[:, c.feature_idx], c.threshold)
        lp = 1.0 - rp
        left_counts = ((node_weight * lp)[:, None] * y_onehot).sum(axis=0)
        right_counts = ((node_weight * rp)[:, None] * y_onehot).sum(axis=0)
        scores[idx] = mgi(left_counts) + mgi(right_counts)
        gates.append(rp)
        left_counts_list.append(left_counts)
        right_counts_list.append(right_counts)
    return gates, left_counts_list, right_counts_list, scores


def train_soft_mgi(
    X: np.ndarray,
    y_onehot: np.ndarray,
    candidates: list[PublicSplitCandidate],
    depth: int,
    beta: float,
    score_normalizer: float | None = None,
    normalizer_mode: str = "fixed",
) -> SoftMgiNode:
    """soft-MGI로 fixed-depth tree 학습 (client decrypt/argmin 없음, plaintext 시뮬레이션).

    normalizer_mode="fixed": norm_score = score / n_samples**2 (모든 노드 공통 공개 상수).
      depth=1에서는 잘 되지만, depth>=2에서는 beta를 어지간히 올려도(50 정도로는 부족,
      100~2000까지 올려야 hard-argmin을 따라잡음 - 2026-08-06 plaintext sweep 확인) 노드가
      깊어질수록 후보 간 점수 차이가 score_normalizer(n_samples**2, 고정)에 비해 상대적으로
      작아져서 softmax가 계속 완만해지기 때문. beta를 그만큼 올리면 실제 CKKS
      exp(-beta*x) 다항식 근사가 못 버틴다(exp_approx_coeffs.py 참고, beta>=100은
      cheb2poly 자체가 수치적으로 붕괴).
    normalizer_mode="range": norm_score = (score-min)/(max-min) (그 노드의 후보 K개 점수
      범위로 동적 정규화, 항상 [0,1]). 후보 간 상대적 차이가 노드 깊이와 무관하게 항상 꽉 찬
      [0,1] 범위로 펴져서, beta<=50(실제 CKKS 근사가 지원하는 범위)만으로도 depth<=3에서
      hard-argmin과 동등하거나 오히려 더 나은 정확도가 나옴(2026-08-06 확인). 단, min/max
      자체를 encrypted 상태에서 구하는 게 아직 미해결 - candidate 수(K=12)가 작아서
      sign_bootstrap 기반 min/max보다 훨씬 싼 대안(예: LogSumExp 기반 soft-min, K개짜리라
      비용 작음)이 있을 가능성이 있음, CKKS 포팅 시 검토 필요.
    """
    if normalizer_mode not in ("fixed", "range"):
        raise ValueError(f"unknown normalizer_mode: {normalizer_mode!r}")
    if score_normalizer is None:
        score_normalizer = float(X.shape[0] ** 2)

    def build(node_weight: np.ndarray, current_depth: int) -> SoftMgiNode:
        if current_depth == depth:
            counts = (node_weight[:, None] * y_onehot).sum(axis=0)
            return SoftMgiLeafNode(counts=counts)

        gates, left_counts_list, right_counts_list, scores = _evaluate_candidates(
            X, y_onehot, candidates, node_weight
        )
        if normalizer_mode == "fixed":
            norm_scores = scores / score_normalizer
        else:
            lo, hi = scores.min(), scores.max()
            norm_scores = (scores - lo) / max(hi - lo, 1e-9)
        w = np.exp(-beta * norm_scores)
        w = w / w.sum()

        blended_gate = sum(w[i] * gates[i] for i in range(len(candidates)))
        left_child_weight = node_weight * (1.0 - blended_gate)
        right_child_weight = node_weight * blended_gate

        left_node = build(left_child_weight, current_depth + 1)
        right_node = build(right_child_weight, current_depth + 1)
        return SoftMgiSplitNode(weights=w, scores=scores, left=left_node, right=right_node)

    return build(np.ones(X.shape[0]), current_depth=0)


def train_hard_mgi_argmin(
    X: np.ndarray,
    y_onehot: np.ndarray,
    candidates: list[PublicSplitCandidate],
    depth: int,
) -> SoftMgiHardNode:
    """비교용 baseline: 매 노드에서 MGI가 최소인 candidate 1개만 고른다 (argmin, blend 없음).
    soft-MGI와 정확히 같은 candidate 풀/재귀 구조를 쓰고 '고르는 방식'만 다르게 해서,
    순수하게 soft vs hard의 효과만 비교할 수 있게 한다."""

    def build(node_weight: np.ndarray, current_depth: int) -> SoftMgiHardNode:
        if current_depth == depth:
            counts = (node_weight[:, None] * y_onehot).sum(axis=0)
            return SoftMgiLeafNode(counts=counts)

        gates, left_counts_list, right_counts_list, scores = _evaluate_candidates(
            X, y_onehot, candidates, node_weight
        )
        winner_idx = int(np.argmin(scores))
        rp = gates[winner_idx]
        left_child_weight = node_weight * (1.0 - rp)
        right_child_weight = node_weight * rp

        left_node = build(left_child_weight, current_depth + 1)
        right_node = build(right_child_weight, current_depth + 1)
        return SoftMgiHardBaselineSplitNode(
            winner_idx=winner_idx, scores=scores, left=left_node, right=right_node
        )

    return build(np.ones(X.shape[0]), current_depth=0)


def _predict_scores(X: np.ndarray, node, candidates, node_weight: np.ndarray, out_scores: np.ndarray) -> None:
    if isinstance(node, SoftMgiLeafNode):
        out_scores += node_weight[:, None] * node.counts[None, :]
        return
    if isinstance(node, SoftMgiSplitNode):
        gates = [sigmoid_gate(X[:, c.feature_idx], c.threshold) for c in candidates]
        blended_gate = sum(node.weights[i] * gates[i] for i in range(len(candidates)))
    else:  # SoftMgiHardBaselineSplitNode
        c = candidates[node.winner_idx]
        blended_gate = sigmoid_gate(X[:, c.feature_idx], c.threshold)
    _predict_scores(X, node.left, candidates, node_weight * (1.0 - blended_gate), out_scores)
    _predict_scores(X, node.right, candidates, node_weight * blended_gate, out_scores)


def predict(X: np.ndarray, root, candidates, n_classes: int) -> np.ndarray:
    scores = np.zeros((X.shape[0], n_classes))
    _predict_scores(X, root, candidates, np.ones(X.shape[0]), scores)
    return np.argmax(scores, axis=1)
