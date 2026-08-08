"""학습 결과로 나오는 tree 모델과 client-side selection 기록을 표현하는 dataclass들."""

from __future__ import annotations

from dataclasses import dataclass

from client_assisted.client_ops import ClientSelectedSplit, EncryptedSelectedSplit


@dataclass
class EncryptedFixedDepthTreeModel:
    """server가 encrypted node weights로 만든 fixed-depth tree model.

    node_splits는 pre-order 순서의 encrypted split 목록이다 (ClientFixedDepthSelections.selections와
    같은 순서). encrypted inference에서 root->leaf traversal에 필요하다.
    """

    depth: int
    node_splits: list[EncryptedSelectedSplit]
    leaf_counts: list[list]


@dataclass
class ClientFixedDepthSelections:
    """client가 pre-order 순서로 선택한 fixed-depth split 정보."""

    depth: int
    selections: list[ClientSelectedSplit]
