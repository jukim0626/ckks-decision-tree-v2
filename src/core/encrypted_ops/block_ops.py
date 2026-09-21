"""Decision Tree 모델 구조와 무관한 범용 CKKS block SIMD primitive.

여러 개의 독립된 값(또는 값 묶음)을 하나의 wide ciphertext 안에 block 단위로 나눠 담고
(block k는 슬롯 [k*block_size, (k+1)*block_size) 구간을 쓴다), block별 reduction/추출을
전체 slot_count가 아니라 block_size 폭으로만 수행해서 rotate 횟수를 줄이는 기법이다. "몇
개의 무엇을 block에 담는지"는 전부 호출부가 정하므로 이 파일 자체는 어떤 모델의 tree
구조/gate 계산/파라미터 의미도 모른다.

2026-09-21 1차 리팩터: 이 4개 함수가 `models/gradient_soft_tree/packed/block_ops.py`
(feature-axis 용도)와 `local_loss/tree_ops.py`·`local_loss/tree_ops_packed.py`
(virtual-leaf-axis 용도)에 완전히 동일한 코드로 3번 중복 구현돼 있었다(원래 각 파일이
"이 실험 범위 밖으로 안 퍼지도록 로컬로 재구현했다"는 이유로 일부러 복붙한 것). rotate
방향/횟수, mask 값, 연산 순서 전부 원본 3곳과 동일하게 그대로 옮겼을 뿐 수치 동작은
바뀌지 않았다."""

from __future__ import annotations

import numpy as np

from core.encrypted_ops.slot_packing import next_power_of_two


def compute_block_size(n_samples: int) -> int:
    """B = next_power_of_two(2*n_samples) - block_local_sum 정확성 조건(B/2>=n_samples,
    이웃 block의 패딩 영역이 회전 거리보다 넓어서 실제 데이터가 절대 안 섞임)을 항상
    만족하는 최소 2의 거듭제곱."""
    return next_power_of_two(2 * n_samples)


def scatter_to_blocks(ctx, cts: list, block_size: int):
    """cts[k](슬롯 [0,n_samples)에 실값, 나머지 0)를 block k(슬롯 [k*block_size,
    (k+1)*block_size))로 옮겨서 전부 더한 ciphertext 하나를 반환. rotate(ct,key,shift)[i]
    =ct[i-shift] 관례상, block k로 옮기려면 shift=+k*block_size. 각 cts[k]의 실값 구간은
    rotate 후 block k 안에만 놓이고 block끼리 절대 안 겹친다(len(cts)*block_size<=slot_count는
    호출부 책임 - `assert_layout_fits` 참고)."""
    packed = None
    for k, ct in enumerate(cts):
        piece = ct if k == 0 else ctx.engine.rotate(ct, ctx.rotation_key, k * block_size)
        packed = piece if packed is None else ctx.engine.add(packed, piece)
    return packed


def block_local_sum(ctx, blocked_ct, block_size: int):
    """block마다 로컬 합(=그 block의 실값들의 합)을 구해서, 그 값을 block 안의 모든
    슬롯에 broadcast한 ciphertext를 반환 - `ctx.engine.sum`과 달리 slot_count 전체가
    아니라 block_size 폭으로만 reduction한다(log2(block_size)회 rotate, 서로 다른 block은
    절대 안 섞임 - block_size가 compute_block_size의 B>=2*n_samples 조건을 만족한다는 게
    전제).

    **주의**: 이 결과에서 의미 있는 값은 슬롯 0, block_size, 2*block_size, ...(각 block의
    시작 슬롯)뿐이다 - 다른 슬롯은 partial sliding-window 값이라 절대 직접 읽으면 안 된다
    (`gather_block_tops`로만 추출할 것)."""
    cur = blocked_ct
    s = 1
    while s < block_size:
        cur = ctx.engine.add(cur, ctx.engine.rotate(cur, ctx.rotation_key, -s))
        s *= 2
    return cur


def gather_block_tops(ctx, reduced_ct, n_blocks: int, block_size: int, slot_count: int):
    """`block_local_sum` 결과에서 각 block의 시작 슬롯(k*block_size)에 있는 유효값만 뽑아서
    슬롯 0..n_blocks-1에 packing. rotate(ct,key,shift)[i]=ct[i-shift] 관례상
    new[k]=old[k*block_size]가 되려면 shift=-k*(block_size-1)
    (i=k, i-shift=k*block_size -> shift=k-k*block_size=-k*(block_size-1))."""
    packed = None
    for k in range(n_blocks):
        mask = np.zeros(slot_count)
        mask[k * block_size] = 1.0
        masked = ctx.engine.multiply(reduced_ct, mask)
        piece = masked if k == 0 else ctx.engine.rotate(masked, ctx.rotation_key, -k * (block_size - 1))
        packed = piece if packed is None else ctx.engine.add(packed, piece)
    return packed
