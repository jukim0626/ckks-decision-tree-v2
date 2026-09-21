"""block_packed_softmax(block_ops.py)이 실제 desilofhe CKKS 엔진(CPU mode)에서
plaintext softmax와 일치하는지 검증하는 self-test. test_block_ops.py와 같은 원칙:
GPU/bootstrap 없이 CPU mode로 먼저 정합성부터 확인한다.

2026-09-15에 numpy(np.roll rotate 시뮬레이션)로만 검증됐던 걸, 여기서 처음으로 실제
CKKS 엔진(다항식 evaluate_polynomial, bounded Newton-Raphson reciprocal 포함)으로 확인한다.

python -m models.gradient_soft_tree.packed.test_block_packed_softmax
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from desilofhe import Engine

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from models.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    compute_block_size,
    block_packed_softmax,
)
from core.encrypted_ops.softmax import SOFTMAX_EXP_INTERVAL  # noqa: E402

TOL = 1e-3


def _make_engine(slot_count: int, max_level: int):
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


def _check(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    err = np.abs(actual - expected).max()
    status = "OK" if err < TOL else "FAIL"
    print(f"[{status}] {name}  max_abs_err={err:.6f}")
    if status == "FAIL":
        raise AssertionError(f"{name} 실패: max_abs_err={err}")


def _run_case(n_groups: int, n_valid: int, slot_count: int, max_level: int, seed: int) -> None:
    block_size = compute_block_size(n_valid)
    assert n_groups * block_size <= slot_count, "toy case 레이아웃이 slot_count를 넘음"
    ctx = _make_engine(slot_count, max_level)

    rng = np.random.default_rng(seed)
    lo, hi = SOFTMAX_EXP_INTERVAL
    margin = 0.1
    vals = rng.uniform(lo + margin, hi - margin, size=(n_groups, n_valid))

    wide = np.zeros(slot_count)
    for g in range(n_groups):
        wide[g * block_size: g * block_size + n_valid] = vals[g]
    ct = ctx.engine.encrypt(wide.tolist(), ctx.pk)

    w = block_packed_softmax(ctx, ct, n_groups, n_valid, block_size, reciprocal_iterations=10, min_level=5)
    dec = np.real(ctx.engine.decrypt(w, ctx.sk))

    exp_v = np.exp(vals)
    expected = exp_v / exp_v.sum(axis=1, keepdims=True)

    for g in range(n_groups):
        got = dec[g * block_size: g * block_size + n_valid]
        _check(f"n_groups={n_groups} n_valid={n_valid} group{g}", got, expected[g])
        pad = dec[g * block_size + n_valid: (g + 1) * block_size]
        _check(f"n_groups={n_groups} n_valid={n_valid} group{g} padding-zero", pad, np.zeros_like(pad))


def main() -> None:
    # toy: gate.py의 attention softmax(node별, n_features 축)를 흉내 - node 3개, feature 4개
    _run_case(n_groups=3, n_valid=4, slot_count=128, max_level=30, seed=0)

    # wine 규모: depth=3 attention softmax(internal node 7개, n_features=13)를 흉내
    n_features = 13
    block_size_wine = compute_block_size(n_features)
    print(f"\n[wine-scale attention] n_internal=7, n_features={n_features}, block_size={block_size_wine}")
    _run_case(n_groups=7, n_valid=n_features, slot_count=2048, max_level=30, seed=1)

    print("\n모든 block_packed_softmax self-test 통과 (실제 CKKS 엔진, CPU mode).")


if __name__ == "__main__":
    main()
