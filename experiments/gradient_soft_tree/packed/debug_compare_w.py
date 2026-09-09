"""threshold 경로에서만 쓰는 패턴("packed 전체를 곱한 뒤 나중에 slot별로 추출")이
baseline이 쓰는 패턴("slot별로 먼저 추출한 뒤 곱하기")과 정말 같은 값을 내는지 직접 검증.
w(packed_softmax 결과)와 sum_t_packed(가상의 packed 벡터, 여기선 sample_mask에서 유도한
간단한 값)를 갖고 두 순서를 비교한다."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from client_assisted.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from closed_form_mgi.primitives import create_bootstrap_context, ensure_level  # noqa: E402
from closed_form_mgi.simd_argmin import next_power_of_two  # noqa: E402
from closed_form_mgi.soft_mgi import _extract_weight_broadcast  # noqa: E402
from experiments.gradient_soft_tree.depth1_ckks import packed_softmax  # noqa: E402
from experiments.gradient_soft_tree.depthN_ckks import init_encrypted_params_N  # noqa: E402

_LOCAL_MIN_LEVEL = 5


def main():
    dataset_name, depth, seed, level_preset = "iris", 1, 0, 17
    X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name)
    n_features = X_train.shape[1]
    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    y_train_oh = one_hot_encode(y_train, n_classes)
    n_pow2_f = next_power_of_two(n_features)

    ctx = create_bootstrap_context(mode="gpu", level_preset=level_preset)
    params = init_encrypted_params_N(ctx, n_features, n_classes, depth, seed=seed, slot_count=ctx.engine.slot_count)
    i = 0
    w = packed_softmax(ctx, params["alpha"][i], n_features, n_pow2_f)

    print("=== w 전체 슬롯 내용 (0..10) ===")
    dec_w = np.real(ctx.engine.decrypt(w, ctx.sk))
    print(np.round(dec_w[:10], 6))
    print("w 합계(0..n_features-1):", dec_w[:n_features].sum(), " (softmax니까 1이어야 함)")
    print("w[n_features:20] (패딩 구간, iris는 n_pow2_f==n_features라 원래 패딩 없음):", np.round(dec_w[n_features:20], 8))
    print("w 먼 슬롯(5000, 20000):", dec_w[5000], dec_w[20000])

    # 임의의 "가짜 sum_t_packed": slot j에 (j+1)*10.0
    fake_vals = [10.0, 20.0, 30.0, 40.0] + [0.0] * (ctx.engine.slot_count - 4)
    fake_sum_t_packed = ctx.engine.encrypt(fake_vals, ctx.pk)

    print("\n=== 순서 A: baseline 방식(추출 먼저, 그다음 곱) ===")
    for j in range(n_features):
        w_j = ensure_level(ctx, _extract_weight_broadcast(ctx, w, j), min_level=_LOCAL_MIN_LEVEL)
        sum_t_j = ensure_level(ctx, _extract_weight_broadcast(ctx, fake_sum_t_packed, j), min_level=_LOCAL_MIN_LEVEL)
        prod = ctx.engine.multiply(sum_t_j, w_j, ctx.rlk)
        val = np.real(ctx.engine.decrypt(prod, ctx.sk))[0]
        print(f"  feature {j}: {val:.6f}  (기대값={fake_vals[j]*dec_w[j]:.6f})")

    print("\n=== 순서 B: packed 방식(곱 먼저, 그다음 추출) ===")
    w_for_t = ensure_level(ctx, w, min_level=_LOCAL_MIN_LEVEL)
    prod_packed = ctx.engine.multiply(fake_sum_t_packed, w_for_t, ctx.rlk)
    prod_packed = ensure_level(ctx, prod_packed, min_level=_LOCAL_MIN_LEVEL)
    for j in range(n_features):
        val = np.real(ctx.engine.decrypt(_extract_weight_broadcast(ctx, prod_packed, j), ctx.sk))[0]
        print(f"  feature {j}: {val:.6f}  (기대값={fake_vals[j]*dec_w[j]:.6f})")


if __name__ == "__main__":
    main()
