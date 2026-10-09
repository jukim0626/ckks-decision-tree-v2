"""2026-09-30: feature 수가 많아 `n_features*block_size > slot_count`인 데이터셋(예:
soybean, 35 feature)을 위한 **opt-in** 확장 - 기존 `packed/block_ops.py`,
`packed/tree_ops_packed.py`(iris/wine/breast_cancer가 실제로 쓰는 코드)는 이 파일이 존재하기
전과 완전히 동일하게 남아있다. 여기서는 feature 축을 여러 ciphertext "그룹"으로 나눠서,
그룹 하나는 기존 단일-ciphertext 경로와 수학적으로 동일하게 동작하고 그룹이 여러 개면 그
결과를 조합한다 - 그룹 간은 서로 다른 feature라 완전히 독립이므로 안전.

**설계**: 그룹 g는 feature 인덱스 구간 [start_g, end_g)를 담당한다. 그룹 자신의
ciphertext 안에서는 기존과 동일하게 로컬 인덱스 0..len(g)-1로 block에 packing된다(그룹의
"로컬 좌표계"). attention 가중치 `w`(alpha의 softmax, n_features 전체가 이미 하나의 작은
compact ciphertext에 들어있음 - n_pow2_f<=64 수준이라 feature grouping과 무관하게 전혀
문제가 없다)는 전역(global) 좌표계라서, 그룹의 로컬 결과와 결합하려면 딱 한 번의
rotate로 좌표계를 옮겨야 한다(아래 `_extract_w_local`/`_scatter_local_to_global` 참고)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from models.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    build_block_masks,
    pack_dataset_features_blocked,
    pack_threshold_blocked,
)


def plan_feature_groups(n_features: int, block_size: int, slot_count: int) -> list[tuple[int, int]]:
    """feature 인덱스 [0,n_features)를 `max_per_group = slot_count // block_size`씩 잘라
    그룹(start,end) 리스트로 나눈다 - 각 그룹의 ciphertext는 `len(group)*block_size <=
    slot_count`를 항상 만족(마지막 그룹은 더 작을 수 있음). 순수 함수라 CKKS 없이 바로
    테스트 가능."""
    max_per_group = slot_count // block_size
    if max_per_group < 1:
        raise ValueError(f"block_size({block_size})가 slot_count({slot_count})보다 커서 feature 1개도 그룹에 못 담는다.")
    groups = []
    start = 0
    while start < n_features:
        end = min(start + max_per_group, n_features)
        groups.append((start, end))
        start = end
    return groups


def pack_dataset_features_wide(ctx, enc_features: list, block_size: int, groups: list[tuple[int, int]]) -> list:
    """`enc_features`(n_features개)를 그룹별로 나눠서 그룹마다 하나씩(기존
    `pack_dataset_features_blocked`와 동일한 로컬 block 레이아웃으로) 만든다."""
    return [pack_dataset_features_blocked(ctx, enc_features[start:end], block_size) for start, end in groups]


def build_block_masks_local(max_per_group: int, block_size: int, slot_count: int) -> list[np.ndarray]:
    """그룹 로컬 인덱스(0..max_per_group-1) 기준 block mask - 그룹마다 크기가 달라도(마지막
    그룹이 더 작아도) 이 마스크 리스트에서 앞쪽만 잘라 쓰면 되므로 최대 크기로 한 번만
    만든다."""
    return build_block_masks(max_per_group, block_size, slot_count)


def pack_threshold_wide(ctx, threshold_cts: list, block_masks_local: list, groups: list[tuple[int, int]]) -> list:
    """`threshold_cts`(n_features개, feature별 broadcast ciphertext)를 그룹별로 나눠서
    로컬 block 레이아웃으로 packing(기존 `pack_threshold_blocked`를 그룹마다 호출)."""
    return [
        pack_threshold_blocked(ctx, threshold_cts[start:end], block_masks_local[: end - start])
        for start, end in groups
    ]


def sample_mask_blocked_wide(n_samples: int, groups: list[tuple[int, int]], block_size: int, slot_count: int) -> list[np.ndarray]:
    """그룹별 sample_mask_blocked(plaintext) - 각 그룹 폭(len(group))만큼만 block을 채운다
    (`packed/block_ops.py`의 `sample_mask_blocked_plain`과 동일한 값/위치, 그룹 로컬
    좌표계로 재구성)."""
    masks = []
    for start, end in groups:
        n_local = end - start
        mask = np.zeros(slot_count)
        for j in range(n_local):
            mask[j * block_size: j * block_size + n_samples] = 1.0
        masks.append(mask)
    return masks


def extract_w_local(ctx, w, group_start: int):
    """전역(global) 좌표계의 compact `w`(alpha softmax 결과, 슬롯 0..n_features-1)에서
    그룹의 로컬 좌표계(슬롯 0..len(group)-1이 전역 슬롯 group_start..group_end-1에 대응)로
    옮긴다 - `rotate(ct,shift)[i]=ct[i-shift]` 관례상 `w_local[i]=w[i+group_start]`가
    되려면 shift=-group_start. 그룹 폭 밖의 슬롯은 신경 안 써도 된다(이 결과는 항상
    그룹 로컬 폭 안에서만 0이 아닌 값과 곱해지므로 - 호출부 참고)."""
    return ctx.engine.rotate(w, ctx.rotation_key, -group_start) if group_start != 0 else w


def scatter_local_to_global(ctx, ct_local, group_start: int):
    """그룹 로컬 좌표계(슬롯 0..len(group)-1)의 compact 결과를 전역 좌표계(슬롯
    group_start..group_end-1)로 되돌린다 - `extract_w_local`의 역연산
    (`rotate(ct,shift)[i]=ct[i-shift]`, shift=+group_start)."""
    return ctx.engine.rotate(ct_local, ctx.rotation_key, group_start) if group_start != 0 else ct_local
