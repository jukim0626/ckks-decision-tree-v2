"""CKKS 엔진을 흉내 내는 numpy 시뮬레이터 - levelpacked의 슬롯 레이아웃/회전 로직을 GPU 없이 몇 초 만에
검증하기 위함. 암호화/노이즈는 없고 값은 그대로 계산한다(rotate = np.roll). 레벨은 대략적으로만 추적해서
bootstrap이 몇 번 일어날지 감을 잡는 용도로 쓴다(실제 desilofhe 레벨 소모와 정확히 같지는 않다)."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

import numpy as np

SIM_MAX_LEVEL = 12  # bootstrap 직후 레벨(근사값)


class SimCt:
    __slots__ = ("v", "level")

    def __init__(self, v, level=SIM_MAX_LEVEL):
        self.v = np.asarray(v, dtype=float)
        self.level = level


def _val(x):
    return x.v if isinstance(x, SimCt) else x


def _lvl(*xs):
    return min(x.level for x in xs if isinstance(x, SimCt))


class SimEngine:
    def __init__(self, slot_count: int = 32768):
        self.slot_count = slot_count
        self.n_bootstrap = 0

    def encrypt(self, values, pk=None):
        v = np.zeros(self.slot_count)
        values = np.asarray(values, dtype=float)
        v[: len(values)] = values
        return SimCt(v)

    def decrypt(self, ct, sk=None):
        return ct.v.copy()

    def add(self, a, b):
        return SimCt(_val(a) + _val(b), _lvl(a, b))

    def subtract(self, a, b):
        return SimCt(_val(a) - _val(b), _lvl(a, b))

    def multiply(self, a, b, rlk=None):
        return SimCt(_val(a) * _val(b), _lvl(a, b) - 1)

    def rotate(self, ct, key, shift):
        return SimCt(np.roll(ct.v, shift), ct.level)

    def evaluate_polynomial(self, ct, coeffs, rlk=None):
        deg = len(coeffs) - 1
        return SimCt(np.polynomial.polynomial.polyval(ct.v, coeffs), ct.level - ceil(log2(deg + 1)))

    def intt(self, ct):
        return ct

    def bootstrap(self, ct, *keys):
        self.n_bootstrap += 1
        return SimCt(ct.v, SIM_MAX_LEVEL)

    def merge_bootstrap(self, a, b, *keys):
        """ct 두 개를 bootstrap 한 번으로(desilofhe merge_bootstrap) - 호출 1회로 센다."""
        self.n_bootstrap += 1
        return SimCt(a.v, SIM_MAX_LEVEL), SimCt(b.v, SIM_MAX_LEVEL)


@dataclass
class SimContext:
    engine: SimEngine
    sk: object = None
    pk: object = None
    rlk: object = None
    rotation_key: object = None
    conjugation_key: object = None
    small_bootstrap_key: object = None
