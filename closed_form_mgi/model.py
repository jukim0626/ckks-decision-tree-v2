"""closed-form MGI tree의 학습 결과를 담는 모델. client_assisted/models.py의
EncryptedFixedDepthTreeModel(depth, node_splits, leaf_counts) 패턴을 그대로 따르되,
이 방식은 "candidate 하나를 선택"하지 않고 항상 모든 feature의 gate를 blend하므로
node_splits(선택된 split 하나) 대신 노드별 (thresholds, soft-MGI weights)를 저장한다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClosedFormMgiNode:
    """한 노드의 라우팅 정보. 새 dataset(train/test/임의 샘플)의 gate를 재계산하는 데
    필요한 것만 담는다 - thresholds로 feature별 gate를 만들고, weights로 그 gate들을
    blend한다 (둘 다 학습에서 재사용, 새로 학습하지 않음)."""

    thresholds: list  # feature별 encrypted_weighted_threshold 결과, pre-order
    weights: object  # soft_mgi_weights() 결과 (이 노드의 softmax blend 가중치)


@dataclass
class ClosedFormMgiTreeModel:
    depth: int
    nodes: list  # ClosedFormMgiNode, pre-order (내부 노드만, 길이 (1<<depth)-1)
    leaf_counts: list  # 리프별 encrypted class count, pre-order 리프 순서
