"""feature-axis SIMD packing primitives.

`depthN_ckks.py`의 `forward_backward_update_N`은 `for j in range(n_features)` 루프
안에서 노드마다 `sigmoid_approx_enc`를 feature 개수만큼 개별 호출하고, backward의
threshold/attention gradient 계산도 feature마다 개별로 `ctx.engine.sum(...)`(전체
slot_count=32768폭, log2(32768)=15회 rotate)을 부른다. 이 모듈은 그 `n_features`번의
호출을 **1번**으로 합치기 위한 새 CKKS 프리미티브를 제공한다 - `depth1_ckks.packed_softmax`가
이미 이 codebase에서 검증한 "K개 값을 슬롯에 packing -> 다항식 1번 평가 -> reduction"
패턴을 feature-diff/sigmoid 축에도 적용한 것.

**block 레이아웃**: feature j의 값(샘플별, n_samples개)을 슬롯 `[j*B, j*B+n_samples)`에
담고 나머지 `[j*B+n_samples, (j+1)*B)`는 0으로 패딩한다. `B`(block_size)는
`compute_block_size()`가 `next_power_of_two(2*n_samples)`로 고정한다 - 이 조건
(`B/2 >= n_samples`)이 `block_local_sum`의 정확성 조건이다: 이 파일이 쓰는 유일한
reduction(`block_local_sum`)은 doubling rotate-add fold(거리 1,2,4,...,B/2)인데, 최대
회전 거리가 B/2이므로 각 block의 패딩 영역이 B/2 이상만 넓으면 이웃 block의 실제
데이터가 절대 섞여 들어올 수 없다(패딩=0인 영역만 들어온다). CPU 모드 desilofhe로
5-feature/6-sample/B=16 toy case에서 손계산과 정확히 일치함을 확인했다(설계 단계에서
검증, `test_block_ops.py`가 이 검증을 코드로 고정한다).

**패딩 오염 주의**: threshold는 (baseline과 동일하게) 이미 전체 슬롯에 broadcast된
ciphertext라, block으로 마스킹만 해도 block 안의 패딩 영역엔 여전히 threshold 값이
남는다 - `enc_diff`의 패딩 영역은 `0 - threshold_j = -threshold_j`로, sigmoid(-threshold_j)는
0이 아니다. `depth1_ckks.py` 모듈 docstring이 명시한 관례("매 sum 호출 전에 한 번씩
sample_mask를 곱해서 이 오염을 제거")를 그대로 따라, `extract_block_to_full`이
sample_mask 곱셈을 자체적으로 강제한다(호출부가 잊어도 안전하도록) - 2026-08-27
masking 버그(패딩 오염이 2 epoch 지나야 발산으로 드러난 사례)와 같은 종류의 실수를
구조적으로 막기 위함.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from closed_form_mgi.simd_argmin import next_power_of_two  # noqa: E402


def compute_block_size(n_samples: int) -> int:
    """B = next_power_of_two(2*n_samples) - block_local_sum 정확성 조건(B/2>=n_samples)을
    항상 만족하는 최소 2의 거듭제곱."""
    return next_power_of_two(2 * n_samples)


def assert_layout_fits(n_features: int, block_size: int, slot_count: int) -> None:
    total = n_features * block_size
    if total > slot_count:
        raise ValueError(
            f"n_features({n_features}) * block_size({block_size}) = {total} > slot_count({slot_count}) - "
            "이 packing 설계는 iris/wine 규모(n_features<=13)를 위한 것으로, breast_cancer(30)/"
            "digits(64) 같은 넓은 데이터셋은 block_size가 더 작아지도록 별도 설계가 필요하다."
        )


def build_block_masks(n_features: int, block_size: int, slot_count: int) -> list[np.ndarray]:
    """block j 안(슬롯 [j*B,(j+1)*B))만 1, 나머지 0인 평문 마스크. n_features/block_size의
    순수 함수라 노드/epoch과 무관하게 한 번만 만들어서 재사용한다."""
    masks = []
    for j in range(n_features):
        m = np.zeros(slot_count)
        m[j * block_size : j * block_size + block_size] = 1.0
        masks.append(m)
    return masks


def _scatter_features_to_blocks(ctx, cts: list, block_size: int):
    """cts[j](슬롯 [0,n_samples)에 실값, 나머지 0)를 block j(슬롯 [j*B,(j+1)*B))로 옮겨서
    전부 더한 ciphertext 하나를 반환. rotate(ct,key,shift)[i]=ct[i-shift] 관례상, block j로
    옮기려면 shift=+j*B(simd_argmin.scatter_to_slot/simd_reduce_argmin과 같은 부호 관례).
    각 cts[j]의 실값 구간은 rotate 후 block j 안에만 놓이고 block끼리 절대 안 겹친다
    (n_features*block_size<=slot_count을 assert_layout_fits로 이미 보장)."""
    packed = None
    for j, ct in enumerate(cts):
        piece = ct if j == 0 else ctx.engine.rotate(ct, ctx.rotation_key, j * block_size)
        packed = piece if packed is None else ctx.engine.add(packed, piece)
    return packed


def pack_dataset_features_blocked(ctx, enc_features: list, block_size: int):
    """dataset.enc_features(n_features개 개별 ciphertext, 각 슬롯 [0,n_samples)에 실값)를
    block 레이아웃 하나로 합친다. 데이터셋은 epoch마다 안 바뀌므로 setup 시 1회만 호출."""
    return _scatter_features_to_blocks(ctx, enc_features, block_size)


def broadcast_full_to_blocks(ctx, full_ct, n_features: int, block_size: int):
    """샘플축 ciphertext(슬롯 [0,n_samples)에 실값) 하나를 모든 block에 복제. 같은 ct를
    n_features번 반복한 리스트에 _scatter_features_to_blocks를 적용하는 것과 동일 -
    dL_dgate_i/gate(노드·epoch마다), sample_mask(setup 시 1회)에 쓴다."""
    return _scatter_features_to_blocks(ctx, [full_ct] * n_features, block_size)


def pack_threshold_blocked(ctx, threshold_cts: list, block_masks: list):
    """threshold_cts[j](이미 전체 슬롯에 broadcast된 ciphertext - 회전해도 값이 안 바뀜)를
    block마다 마스킹해서 합친다. rotate가 아니라 masking을 쓰는 이유: threshold는 이미
    회전-불변(모든 슬롯이 같은 값)이라 회전으로는 block에 못 가둔다. 노드·epoch마다 호출
    (threshold는 매 epoch 갱신됨)."""
    packed = None
    for j, (ct, mask) in enumerate(zip(threshold_cts, block_masks)):
        piece = ctx.engine.multiply(ct, mask)
        packed = piece if packed is None else ctx.engine.add(packed, piece)
    return packed


def block_local_sum(ctx, blocked_ct, block_size: int):
    """block마다 로컬 합(=그 block의 n_samples개 실값의 합)을 구해서, 그 값을 block 안의
    모든 슬롯에 broadcast한 ciphertext를 반환 - `ctx.engine.sum`과 달리 slot_count 전체가
    아니라 block_size 폭으로만 reduction한다(log2(block_size)회 rotate, 서로 다른 block은
    절대 안 섞임 - 모듈 docstring의 B>=2*n_samples 조건 참고).

    **주의**: 이 결과에서 의미 있는 값은 슬롯 0, block_size, 2*block_size, ...(각 block의
    시작 슬롯)뿐이다 - 다른 슬롯은 partial sliding-window 값이라 절대 직접 읽으면 안 된다
    (`gather_block_tops_to_packed`로만 추출할 것)."""
    cur = blocked_ct
    s = 1
    while s < block_size:
        cur = ctx.engine.add(cur, ctx.engine.rotate(cur, ctx.rotation_key, -s))
        s *= 2
    return cur


def gather_block_tops_to_packed(ctx, reduced_ct, n_features: int, block_size: int, slot_count: int):
    """block_local_sum 결과에서 각 block의 시작 슬롯(j*block_size)에 있는 유효값만 뽑아서
    슬롯 0..n_features-1에 packing - `w`(packed_softmax 결과)와 같은 레이아웃이라, 이후
    dL_dt_j 전부를 `w`와의 elementwise 곱 한 번으로 계산할 수 있게 해준다.
    rotate(ct,key,shift)[i]=ct[i-shift] 관례상 new[j]=old[j*B]가 되려면 shift=-j*(B-1)
    (i=j, i-shift=j*B -> shift=j-j*B=-j*(B-1))."""
    packed = None
    for j in range(n_features):
        mask = np.zeros(slot_count)
        mask[j * block_size] = 1.0
        masked = ctx.engine.multiply(reduced_ct, mask)
        piece = masked if j == 0 else ctx.engine.rotate(masked, ctx.rotation_key, -j * (block_size - 1))
        packed = piece if packed is None else ctx.engine.add(packed, piece)
    return packed


def extract_block_to_full(ctx, blocked_ct, feature_idx: int, block_size: int, sample_mask):
    """block feature_idx(슬롯 [j*B,(j+1)*B))를 슬롯 0 기준 full-width 레이아웃으로 되돌리고,
    **항상** sample_mask를 곱한다(호출부가 빼먹을 수 없도록 강제).

    baseline(depthN_ckks.py)의 forward `gate_j = sigmoid_approx_enc(...)`는 마스킹 없이 바로
    `w_j*gate_j` 가중합에 쓰이는데, 그래도 안전한 이유는 baseline의 `enc_feature`가 회전 한
    번도 없이 encrypt 시점부터 그대로인 단일 ciphertext라 n_samples 밖이 원래부터 깨끗한
    0이기 때문이다. **packed 버전은 이 전제가 깨진다**: `extract_block_to_full`이 돌려주는
    값은 `gate_blocked`(폭 n_features*block_size 밖은 diff=0-0이라 sigmoid(0)=0.5로 채워짐,
    그 안도 다른 feature block의 padding이 rotate로 섞여 들어올 수 있음)를 **회전**해서 얻은
    것이라 애초에 n_samples 밖이 깨끗하지 않다 - 여기서 마스킹을 빼면 `gate`(및 이걸로
    backward에서 만드는 `gate_broadcast_blocked`)가 오염된 채로 전파된다. 2026-09 디버깅에서
    실측 확인: 이 마스킹을 빼고 baseline과 1 epoch만 직접 비교했더니 leaf_logits(마스킹과
    무관한 코드)는 거의 일치(diff~1e-5)했지만 alpha/threshold(이 함수를 거치는 경로)는
    이미 0.003~0.009 어긋났다 - 마스킹을 되살리자 정상 범위로 돌아옴. rotate(ct,key,shift)[i]
    =ct[i-shift]로 new[i]=old[i+j*B]가 되려면 shift=-j*B."""
    shifted = blocked_ct if feature_idx == 0 else ctx.engine.rotate(blocked_ct, ctx.rotation_key, -feature_idx * block_size)
    return ctx.engine.multiply(shifted, sample_mask, ctx.rlk)
