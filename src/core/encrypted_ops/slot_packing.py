"""SIMD 슬롯 packing/추출 프리미티브. 여러 값을 슬롯 여러 개에 packing한 ciphertext에서
특정 슬롯 하나만 뽑아 전체 슬롯에 broadcast하거나, 그 반대(broadcast 값을 슬롯 하나로
모으기)를 한다 - packed_softmax, gradient backward의 alpha/threshold gradient 등에서
반복적으로 쓰이는 패턴.

2026-09-09 리팩터로 closed_form_mgi/simd_argmin.py, soft_mgi.py에서 분리됨."""

from __future__ import annotations

import numpy as np


def next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def scatter_to_slot(ctx, ct, slot_idx: int):
    """scalar 값(모든 슬롯에 broadcast돼 있음 - engine.sum()이 slot 0뿐 아니라 전체
    슬롯에 합계를 복제하기 때문)을 slot_idx 하나에만 남기고 나머지는 0으로 지운 뒤 옮긴다."""
    mask = np.zeros(ctx.engine.slot_count)
    mask[0] = 1.0
    masked = ctx.engine.multiply(ct, mask)
    if slot_idx == 0:
        return masked
    return ctx.engine.rotate(masked, ctx.rotation_key, slot_idx)


def extract_weight_broadcast(ctx, weights, candidate_idx: int):
    """weights의 슬롯 candidate_idx에 있는 값만 남기고 전체 슬롯에 broadcast (다른
    슬롯 레이아웃과 곱하려면 스칼라처럼 모든 슬롯에 같은 값이 있어야 함). scatter_to_slot의
    역방향 연산과 같은 마스크->sum 패턴."""
    mask = np.zeros(ctx.engine.slot_count)
    mask[candidate_idx] = 1.0
    isolated = ctx.engine.multiply(weights, mask)
    isolated = ctx.engine.intt(isolated)
    return ctx.engine.sum(isolated, ctx.rotation_key)
