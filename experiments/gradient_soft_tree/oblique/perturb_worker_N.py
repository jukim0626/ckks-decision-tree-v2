"""2026-09-09 롤백 메커니즘의 "막힘(stuck state)" 문제 대응: 같은 checkpoint에서 같은 lr로
재시도하면 gradient 계산이 결정론적이라 매번 똑같이 실패한다(iris depth=2 lr=6.0 세션에서
epoch13/14/15가 완전히 동일한 값으로 반복 실패하는 게 실측 확인됨). 이 워커는 "이 epoch
포기" 이후, 다음 epoch이 같은 실패를 반복하지 않도록 checkpoint의 w/b에 작은 랜덤 섭동을
더한다(leaf_logits는 안 건드림 - gate 안정성이 목표라 gate 파라미터만).

CKKS 관점에서 안전한 이유: 그냥 공개된 작은 상수(plaintext 랜덤값)를 ciphertext에 더하는
것뿐이라 reciprocal/비교 없음, 곱셈 depth도 안 씀(순수 덧셈)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from closed_form_mgi.io_utils import load_context  # noqa: E402
from closed_form_mgi.primitives import create_bootstrap_engine  # noqa: E402


def main() -> None:
    session_dir = Path(sys.argv[1])
    seed = int(sys.argv[2])
    noise_std = float(sys.argv[3])
    config = json.loads((session_dir / "config.json").read_text())
    n_features = config["n_features"]
    depth = config["depth"]
    n_internal = (1 << depth) - 1

    engine = create_bootstrap_engine(
        mode=config["mode"], device_id=config["device_id"], level_preset=config.get("level_preset")
    )
    ctx = load_context(engine, session_dir / "keys", mode=config["mode"], device_id=config["device_id"])

    rng = np.random.default_rng(seed)
    params_dir = session_dir / "params"
    for i in range(n_internal):
        for j in range(n_features):
            path = params_dir / f"w_{i}_{j}.ct"
            ct = engine.read_ciphertext(path)
            noise = float(rng.normal(0, noise_std))
            ct = engine.add(ct, noise)
            engine.write_ciphertext(ct, path)
        path = params_dir / f"b_{i}.ct"
        ct = engine.read_ciphertext(path)
        noise = float(rng.normal(0, noise_std))
        ct = engine.add(ct, noise)
        engine.write_ciphertext(ct, path)


if __name__ == "__main__":
    main()
