"""server_ops + client_ops + models을 엮어서 실제 encrypted tree를 만드는 orchestration."""

from __future__ import annotations

import gc

from tqdm import tqdm

from client_assisted.candidates import PublicSplitCandidate
from client_assisted.client_ops import (
    ClientSelectedSplit,
    client_encrypt_selected_split,
    client_score_aggregate,
)
from client_assisted.context import EncryptedTrainingContext, sync_engine_if_needed
from client_assisted.dataset import EncryptedDataset
from client_assisted.models import (
    ClientFixedDepthSelections,
    EncryptedFixedDepthTreeModel,
)
from client_assisted.server_ops import (
    encrypt_constant_vector,
    encrypted_class_counts,
    server_compute_one_weighted_candidate_aggregate,
    server_compute_weighted_child_weights,
)


def train_client_assisted_fixed_depth_tree(
    ctx: EncryptedTrainingContext,
    dataset: EncryptedDataset,
    candidates: list[PublicSplitCandidate],
    depth: int,
    verbose: bool = True,
) -> tuple[EncryptedFixedDepthTreeModel, ClientFixedDepthSelections]:
    """논문식 client-assisted 방식으로 fixed-depth encrypted tree를 학습.

    verbose=True면 tqdm 진행률 바로 node 학습 진행 상황을 보여주고, node마다 선택된
    split을 tqdm.write()로 즉시 출력한다 (백그라운드로 파일에 리다이렉트해도 끝날 때까지
    기다리지 않고 중간 로그를 바로 볼 수 있음).
    """
    if depth < 1:
        raise ValueError("depth must be at least 1")

    total_nodes = (1 << depth) - 1  # full binary tree의 internal node 개수 = 2^depth - 1
    selected_splits: list[ClientSelectedSplit] = []
    node_splits = []
    encrypted_leaf_weights = []
    root_weights = encrypt_constant_vector(ctx, 1.0, dataset.n_samples)

    progress = tqdm(total=total_nodes, desc="[training] nodes", disable=not verbose)

    def train_node(enc_node_weights, current_depth: int) -> None:
        if current_depth == depth:
            encrypted_leaf_weights.append(enc_node_weights)
            return

        best_index = -1
        best_score = None
        for idx, candidate in enumerate(candidates):
            aggregate = server_compute_one_weighted_candidate_aggregate(
                ctx,
                dataset,
                candidate,
                enc_node_weights,
            )
            score = client_score_aggregate(ctx, aggregate)
            del aggregate
            if best_score is None or score < best_score:
                best_score = score
                best_index = idx
        best_candidate = candidates[best_index]
        selected = ClientSelectedSplit(
            candidate_index=best_index,
            feature_idx=best_candidate.feature_idx,
            threshold=best_candidate.threshold,
            gini_score=best_score,
        )
        selected_splits.append(selected)
        enc_selected = client_encrypt_selected_split(
            ctx,
            selected,
            dataset.n_samples,
            dataset.n_features,
        )
        node_splits.append(enc_selected)
        if verbose:
            tqdm.write(
                f"[training] node {len(selected_splits)}/{total_nodes} | "
                f"feature={selected.feature_idx} | threshold={selected.threshold:.6f} | "
                f"gini={selected.gini_score:.6f}"
            )
            progress.update(1)
        left_weights, right_weights = server_compute_weighted_child_weights(
            ctx,
            dataset,
            enc_selected,
            enc_node_weights,
        )
        del enc_node_weights
        sync_engine_if_needed(ctx)
        gc.collect()
        train_node(left_weights, current_depth + 1)
        del left_weights
        train_node(right_weights, current_depth + 1)

    train_node(root_weights, current_depth=0)
    progress.close()
    leaf_counts = [
        encrypted_class_counts(ctx, leaf_weights, dataset.enc_labels)
        for leaf_weights in encrypted_leaf_weights
    ]
    model = EncryptedFixedDepthTreeModel(depth=depth, node_splits=node_splits, leaf_counts=leaf_counts)
    selections = ClientFixedDepthSelections(depth=depth, selections=selected_splits)
    return model, selections
