"""Client-assisted encrypted Gini training package.

논문식 client-assisted encrypted training (임의 depth, fixed full binary tree) 구현이다.

Server:
    encrypted data로 candidate별 left/right class aggregate를 계산한다 (server_ops.py).

Client/KMS:
    aggregate만 decrypt하고 weighted Gini + argmin을 수행한다 (client_ops.py).
    selected feature one-hot과 selected threshold를 다시 encrypt해서 server에 보낸다.

Server:
    encrypted selected split으로 child weights와 leaf counts를 계산한다 (server_ops.py).

training.py가 위 두 역할을 엮어서 실제 tree를 만들고, models.py가 그 결과 구조를 표현한다.
verification.py는 protocol core가 아니라 client가 이미 아는 평문 정보로 정확도를 검증하는
전용 helper들이다. inference.py는 depth와 무관한 encrypted inference 공용 helper
(sample encrypt, 최종 score decrypt+argmax)다.

`train_client_assisted_fixed_depth_tree(depth=N)`로 임의 depth 학습 + `main.py`의
`encrypted_traverse_and_predict_fixed_depth()`로 decrypt 없는 root->leaf encrypted
inference가 end-to-end로 검증됐다 (depth=2/3 둘 다, EXPERIMENT_LOG.md 참고).
`EncryptedFixedDepthTreeModel.node_splits`가 각 node의 encrypted split을 pre-order로
보관하고 있어서 가능하다.

depth=2 전용 구현(`EncryptedDepth2TreeModel`, `train_client_assisted_depth2_tree` 등)은
일반화된 fixed-depth 버전이 완전히 대체해서 삭제했다. depth=1 stump와 그 데모
(experiments.py)도 정리하면서 삭제했다.
"""

from __future__ import annotations

from client_assisted.candidates import (
    PublicSplitCandidate,
    make_public_grid_candidates,
    make_small_public_threshold_grid,
)
from client_assisted.client_ops import (
    ClientSelectedSplit,
    EncryptedSelectedSplit,
    client_decrypt_aggregate_counts,
    client_decrypt_scalar,
    client_encrypt_selected_split,
    client_score_aggregate,
    weighted_gini_from_counts,
)
from client_assisted.context import (
    EncryptedTrainingContext,
    create_context,
    sync_engine_if_needed,
)
from client_assisted.dataset import (
    EncryptedDataset,
    encrypt_dataset,
    load_scaled_dataset_subset,
    one_hot_encode,
)
from client_assisted.inference import (
    debug_decrypt_class_scores,
    encrypt_inference_sample,
    encrypted_predict_class,
)
from client_assisted.models import (
    ClientFixedDepthSelections,
    EncryptedFixedDepthTreeModel,
)
from client_assisted.server_ops import (
    EncryptedAggregate,
    encrypt_constant_vector,
    encrypted_class_counts,
    encrypted_slot_sum,
    encrypted_weighted_class_counts,
    server_compute_child_weights,
    server_compute_one_weighted_candidate_aggregate,
    server_compute_weighted_child_weights,
    server_selected_feature_value,
)
from client_assisted.training import train_client_assisted_fixed_depth_tree
from client_assisted.verification import (
    debug_decrypt_fixed_depth_leaf_counts,
    format_float_list,
    plaintext_fixed_depth_leaf_weights,
    plaintext_split_weights,
    predict_fixed_depth_leaf_majority_plaintext,
    predict_fixed_depth_soft_plaintext,
)

__all__ = [
    "PublicSplitCandidate",
    "make_public_grid_candidates",
    "make_small_public_threshold_grid",
    "ClientSelectedSplit",
    "EncryptedSelectedSplit",
    "client_decrypt_aggregate_counts",
    "client_decrypt_scalar",
    "client_encrypt_selected_split",
    "client_score_aggregate",
    "weighted_gini_from_counts",
    "EncryptedTrainingContext",
    "create_context",
    "sync_engine_if_needed",
    "EncryptedDataset",
    "encrypt_dataset",
    "load_scaled_dataset_subset",
    "one_hot_encode",
    "debug_decrypt_class_scores",
    "encrypt_inference_sample",
    "encrypted_predict_class",
    "ClientFixedDepthSelections",
    "EncryptedFixedDepthTreeModel",
    "EncryptedAggregate",
    "encrypt_constant_vector",
    "encrypted_class_counts",
    "encrypted_slot_sum",
    "encrypted_weighted_class_counts",
    "server_compute_child_weights",
    "server_compute_one_weighted_candidate_aggregate",
    "server_compute_weighted_child_weights",
    "server_selected_feature_value",
    "train_client_assisted_fixed_depth_tree",
    "debug_decrypt_fixed_depth_leaf_counts",
    "format_float_list",
    "plaintext_fixed_depth_leaf_weights",
    "plaintext_split_weights",
    "predict_fixed_depth_leaf_majority_plaintext",
    "predict_fixed_depth_soft_plaintext",
]
