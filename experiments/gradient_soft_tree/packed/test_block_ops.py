"""block_ops.py의 새 CKKS 프리미티브를 GPU/bootstrap 없이(CPU mode, plaintext rotation
key만) 검증하는 self-test. 이 codebase의 문서화된 실패 사례(closed_form_mgi/simd_argmin.py
scatter_to_slot 자체 docstring의 "처음엔 rotate만 했는데... 버그가 있었다", EXPERIMENT_LOG.md
2026-08-27의 masking 버그)가 전부 "회전 부호/패딩 오염을 잘못 짚었다"는 종류라, GPU 시간을
쓰기 전에 여기서 손계산과 정확히 일치하는지부터 확인한다.

python -m experiments.gradient_soft_tree.packed.test_block_ops
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from desilofhe import Engine

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    assert_layout_fits,
    block_local_sum,
    broadcast_full_to_blocks,
    build_block_masks,
    compute_block_size,
    extract_block_to_full,
    gather_block_tops_to_packed,
    pack_dataset_features_blocked,
    pack_threshold_blocked,
)

N_FEATURES = 5
N_SAMPLES = 6
TOL = 1e-3


def _make_engine(slot_count: int, max_level: int = 12):
    engine = Engine(slot_count=slot_count, max_level=max_level, mode="cpu", compact=True)
    sk = engine.create_secret_key()
    pk = engine.create_public_key(sk)
    rlk = engine.create_relinearization_key(sk)
    rotation_key = engine.create_rotation_key(sk)

    class Ctx:
        pass

    ctx = Ctx()
    ctx.engine, ctx.sk, ctx.pk, ctx.rlk, ctx.rotation_key = engine, sk, pk, rlk, rotation_key
    return ctx


def _dec(ctx, ct):
    return np.real(ctx.engine.decrypt(ct, ctx.sk))


def _check(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    err = np.abs(actual - expected).max()
    status = "OK" if err < TOL else "FAIL"
    print(f"[{status}] {name}  max_abs_err={err:.6f}")
    if status == "FAIL":
        raise AssertionError(f"{name} 실패: max_abs_err={err}")


def main() -> None:
    block_size = compute_block_size(N_SAMPLES)
    print(f"n_features={N_FEATURES} n_samples={N_SAMPLES} block_size={block_size}")
    assert block_size == 16, f"toy case는 next_power_of_two(2*6)=16을 기대했는데 {block_size}"

    slot_count = 128
    assert_layout_fits(N_FEATURES, block_size, slot_count)
    ctx = _make_engine(slot_count)
    block_masks = build_block_masks(N_FEATURES, block_size, slot_count)

    rng = np.random.default_rng(0)
    feature_vals = rng.normal(0, 1, size=(N_FEATURES, N_SAMPLES))  # feature j, sample s

    # --- pack_dataset_features_blocked: 각 feature를 자기 block에, 서로 안 섞이는지 ---
    enc_features = []
    for j in range(N_FEATURES):
        vec = [0.0] * slot_count
        vec[:N_SAMPLES] = feature_vals[j].tolist()
        enc_features.append(ctx.engine.encrypt(vec, ctx.pk))
    blocked_features = pack_dataset_features_blocked(ctx, enc_features, block_size)
    dec = _dec(ctx, blocked_features)
    expected = np.zeros(slot_count)
    for j in range(N_FEATURES):
        expected[j * block_size : j * block_size + N_SAMPLES] = feature_vals[j]
    _check("pack_dataset_features_blocked", dec, expected)

    # --- block_local_sum: block마다 로컬 합이 block 시작 슬롯에 broadcast ---
    reduced = block_local_sum(ctx, blocked_features, block_size)
    dec = _dec(ctx, reduced)
    per_block_sum = feature_vals.sum(axis=1)  # (n_features,)
    for j in range(N_FEATURES):
        _check(f"block_local_sum block{j} start-slot", dec[j * block_size : j * block_size + 1], per_block_sum[j : j + 1])
    # 이웃 block과 안 섞였는지: block j의 시작 슬롯 값이 다른 feature의 합과 우연히 같지만
    # 않다면(랜덤 데이터라 사실상 항상 다름) 이걸로 교차오염이 없다는 것도 간접 확인된다.

    # --- gather_block_tops_to_packed: block 시작 슬롯들을 0..n_features-1로 모으기 ---
    gathered = gather_block_tops_to_packed(ctx, reduced, N_FEATURES, block_size, slot_count)
    dec = _dec(ctx, gathered)
    _check("gather_block_tops_to_packed", dec[:N_FEATURES], per_block_sum)
    _check("gather_block_tops_to_packed padding-zero", dec[N_FEATURES:N_FEATURES + 4], np.zeros(4))

    # --- broadcast_full_to_blocks: 샘플축 ciphertext 하나를 모든 block에 복제 ---
    sample_vals = rng.normal(0, 1, size=N_SAMPLES)
    full_vec = [0.0] * slot_count
    full_vec[:N_SAMPLES] = sample_vals.tolist()
    full_ct = ctx.engine.encrypt(full_vec, ctx.pk)
    broadcasted = broadcast_full_to_blocks(ctx, full_ct, N_FEATURES, block_size)
    dec = _dec(ctx, broadcasted)
    expected = np.zeros(slot_count)
    for j in range(N_FEATURES):
        expected[j * block_size : j * block_size + N_SAMPLES] = sample_vals
    _check("broadcast_full_to_blocks", dec, expected)

    # --- pack_threshold_blocked: broadcast-전체 threshold를 block마다 마스킹 ---
    thresholds = rng.normal(0, 1, size=N_FEATURES)
    threshold_cts = [ctx.engine.encrypt([float(thresholds[j])] * slot_count, ctx.pk) for j in range(N_FEATURES)]
    blocked_threshold = pack_threshold_blocked(ctx, threshold_cts, block_masks)
    dec = _dec(ctx, blocked_threshold)
    expected = np.zeros(slot_count)
    for j in range(N_FEATURES):
        expected[j * block_size : (j + 1) * block_size] = thresholds[j]
    _check("pack_threshold_blocked", dec, expected)

    # --- extract_block_to_full: block j를 슬롯 0 기준으로 되돌리고 sample_mask로 패딩 오염 제거 ---
    # enc_diff_blocked = blocked_features - blocked_threshold 흉내: block 전체(패딩 포함)에
    # threshold가 남아있는 상태를 재현해서, extract 후 패딩 슬롯이 정확히 0이 되는지 확인.
    enc_diff_blocked = ctx.engine.subtract(blocked_features, blocked_threshold)
    sample_mask_vec = [0.0] * slot_count
    sample_mask_vec[:N_SAMPLES] = [1.0] * N_SAMPLES
    sample_mask = ctx.engine.encrypt(sample_mask_vec, ctx.pk)
    for j in range(N_FEATURES):
        extracted = extract_block_to_full(ctx, enc_diff_blocked, j, block_size, sample_mask)
        dec = _dec(ctx, extracted)
        expected = np.zeros(slot_count)
        expected[:N_SAMPLES] = feature_vals[j] - thresholds[j]
        _check(f"extract_block_to_full feature{j} (real region)", dec[:N_SAMPLES], expected[:N_SAMPLES])
        _check(f"extract_block_to_full feature{j} (padding zeroed)", dec[N_SAMPLES:block_size], np.zeros(block_size - N_SAMPLES))

    print("\n모든 block_ops 프리미티브 self-test 통과 (toy scale, n_features=5/n_samples=6).")


def main_realistic_scale() -> None:
    """wine 규모(n_features=13, n_samples=148, slot_count=32768)에서도 정확한지 확인 -
    toy case(block_size=16)와 달리 block_size=512라 block_local_sum의 rotate 횟수(log2(512)=9)
    등 실제 production 규모의 코드 경로를 그대로 탄다. CPU mode라 rotate 자체가 느려서
    (32768-wide, 수십 초) GPU 시간 없이도 정확성만 먼저 확인하는 용도 - 실제 속도는 Phase A
    프로파일링/Phase C GPU 비교가 답한다."""
    n_features, n_samples = 13, 148
    block_size = compute_block_size(n_samples)
    slot_count = 32768
    assert block_size == 512, f"wine 규모는 next_power_of_two(2*148)=512을 기대했는데 {block_size}"
    assert_layout_fits(n_features, block_size, slot_count)
    print(f"\n[realistic scale] n_features={n_features} n_samples={n_samples} block_size={block_size}")

    ctx = _make_engine(slot_count, max_level=20)  # CPU mode: slot_count=32768엔 max_level>=20 필요(실측)
    block_masks = build_block_masks(n_features, block_size, slot_count)

    rng = np.random.default_rng(1)
    feature_vals = rng.normal(0, 1, size=(n_features, n_samples))
    enc_features = []
    for j in range(n_features):
        vec = [0.0] * slot_count
        vec[:n_samples] = feature_vals[j].tolist()
        enc_features.append(ctx.engine.encrypt(vec, ctx.pk))

    blocked_features = pack_dataset_features_blocked(ctx, enc_features, block_size)
    reduced = block_local_sum(ctx, blocked_features, block_size)
    gathered = gather_block_tops_to_packed(ctx, reduced, n_features, block_size, slot_count)
    dec = _dec(ctx, gathered)
    _check("[realistic] pack+block_local_sum+gather", dec[:n_features], feature_vals.sum(axis=1))

    thresholds = rng.normal(0, 1, size=n_features)
    threshold_cts = [ctx.engine.encrypt([float(thresholds[j])] * slot_count, ctx.pk) for j in range(n_features)]
    blocked_threshold = pack_threshold_blocked(ctx, threshold_cts, block_masks)
    enc_diff_blocked = ctx.engine.subtract(blocked_features, blocked_threshold)
    sample_mask_vec = [0.0] * slot_count
    sample_mask_vec[:n_samples] = [1.0] * n_samples
    sample_mask = ctx.engine.encrypt(sample_mask_vec, ctx.pk)
    for j in range(n_features):
        extracted = extract_block_to_full(ctx, enc_diff_blocked, j, block_size, sample_mask)
        dec = _dec(ctx, extracted)
        expected_real = feature_vals[j] - thresholds[j]
        _check(f"[realistic] extract_block_to_full feature{j} real", dec[:n_samples], expected_real)
        _check(f"[realistic] extract_block_to_full feature{j} padding", dec[n_samples:block_size], np.zeros(block_size - n_samples))

    print("\nwine 규모 self-test 통과.")


if __name__ == "__main__":
    main()
    main_realistic_scale()
