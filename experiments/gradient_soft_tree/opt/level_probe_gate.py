"""degree=3/5/15 gate polynomial이 evaluate_polynomial 한 번에 실제로 몇 level을
소모하는지 실측하는 1회성 프로브 (depthN_ckks.py 주석의 "level_probe2.py 실측" 관행을
그대로 따름). GPU에 데이터를 태우지 않고 최소 연산만 수행 - 몇 초 안에 끝난다.

python -m experiments.gradient_soft_tree.opt.level_probe_gate
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from closed_form_mgi.primitives import create_bootstrap_context, ensure_level  # noqa: E402
from experiments.gradient_soft_tree.opt.poly_gate import gate_coeffs  # noqa: E402


def probe(degree: int, level_preset: int = 17) -> None:
    ctx = create_bootstrap_context(mode="gpu", level_preset=level_preset)
    x = ctx.engine.encrypt([0.3] * ctx.engine.slot_count, ctx.pk)
    x = ensure_level(ctx, x, min_level=level_preset)  # bootstrap해서 최대 level로 맞춤
    level_before = x.level
    y = ctx.engine.evaluate_polynomial(x, gate_coeffs(degree), ctx.rlk)
    level_after = y.level
    consumed = level_before - level_after
    print(f"degree={degree:2d}  level_before={level_before}  level_after={level_after}  consumed={consumed}")


if __name__ == "__main__":
    for d in (15, 5, 3):
        probe(d)
