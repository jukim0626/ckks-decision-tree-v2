"""session_dir의 최종(혹은 현재) 파라미터를 decrypt해서 절댓값 분포를 출력 - CKKS bootstrap이
"ciphertext 값이 2~5 넘으면 조용히 깨진다"는 이 프로젝트의 기존 발견([[project-ckks-decision-tree]]
reciprocal_approx.py 참고)에 해당하는지 확인하는 진단 스크립트.

python -m models.gradient_soft_tree.opt.inspect_param_magnitudes <session_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.serialization import load_context  # noqa: E402
from core.ckks_engine import create_bootstrap_engine  # noqa: E402


def main() -> None:
    session_dir = Path(sys.argv[1])
    config = json.loads((session_dir / "config.json").read_text())
    n_features = config["n_features"]
    depth = config["depth"]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth

    engine = create_bootstrap_engine(mode=config["mode"], device_id=config["device_id"], level_preset=config.get("level_preset"))
    ctx = load_context(engine, session_dir / "keys", mode=config["mode"], device_id=config["device_id"])

    params_dir = session_dir / "params"
    alpha_vals = []
    for i in range(n_internal):
        v = np.real(ctx.engine.decrypt(engine.read_ciphertext(params_dir / f"alpha_{i}.ct"), ctx.sk))[:n_features]
        alpha_vals.append(v)
        print(f"alpha[{i}] = {np.round(v, 4)}")
    threshold_vals = []
    for i in range(n_internal):
        row = []
        for j in range(n_features):
            v = np.real(ctx.engine.decrypt(engine.read_ciphertext(params_dir / f"threshold_{i}_{j}.ct"), ctx.sk))[0]
            row.append(v)
        threshold_vals.append(row)
        print(f"threshold[{i}] = {np.round(row, 4)}")
    leaf_vals = []
    for l in range(n_leaves):
        v = np.real(ctx.engine.decrypt(engine.read_ciphertext(params_dir / f"leaf_{l}.ct"), ctx.sk))
        leaf_vals.append(v)
        print(f"leaf[{l}] = {np.round(v[: config['n_classes']], 4)}  (full padded slots max abs = {np.abs(v).max():.4g})")

    all_abs = np.concatenate([np.abs(np.array(alpha_vals)).ravel(), np.abs(np.array(threshold_vals)).ravel(), np.abs(np.array(leaf_vals)).ravel()])
    print(f"\nmax |value| across all params = {all_abs.max():.6g}")
    print(f"count of |value| > 5 (bootstrap-unsafe zone per project convention): {(all_abs > 5).sum()} / {all_abs.size}")


if __name__ == "__main__":
    main()
