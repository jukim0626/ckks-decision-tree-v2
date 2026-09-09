"""depthN_ckks.forward_backward_update_N의 한 epoch을 별도 프로세스로 실행 (epoch_worker.py를
임의 depth로 일반화 - depth=2 검증에서 실측: 단일 프로세스로 epoch 2에서 CUDA OOM, depth=1과
동일 원인이라 동일 패턴으로 우회).

session_dir/params/alpha_{i}.ct, threshold_{i}_{j}.ct, leaf_{l}.ct를 읽어서 한 epoch
갱신한 뒤 같은 자리에 덮어쓴다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data.serialization import load_context, read_dataset  # noqa: E402
from core.ckks_engine import create_bootstrap_engine  # noqa: E402
from models.gradient_soft_tree.depthN_ckks import forward_backward_update_N  # noqa: E402

# 2026-08-26: 파라미터를 읽자마자 min_level=25로 강제 bootstrap하는 시도를 해봤으나 실패함
# (43개 ciphertext를 전부 즉시 refresh하는 비용 자체가 커서 오히려 더 일찍 OOM남 - desilofhe는
# 프로세스 안에서 메모리가 절대 안 줄어드니 "언제" bootstrap하느냐가 아니라 "총 몇 번"
# bootstrap하느냐가 peak을 결정한다는 걸 확인. depthN_ckks.py의 ensure_level 호출부 min_level
# 자체를 낮춰 bootstrap 총량을 줄이는 방향으로 대신 시도 중 - depthN_ckks.py 상단 주석 참고.


def main() -> None:
    session_dir = Path(sys.argv[1])
    config = json.loads((session_dir / "config.json").read_text())
    n_features = config["n_features"]
    depth = config["depth"]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth

    engine = create_bootstrap_engine(
        mode=config["mode"], device_id=config["device_id"], level_preset=config.get("level_preset")
    )
    ctx = load_context(engine, session_dir / "keys", mode=config["mode"], device_id=config["device_id"])
    dataset = read_dataset(
        ctx,
        session_dir / "dataset",
        n_features=n_features,
        n_classes=config["n_classes"],
        n_samples=config["n_samples"],
    )
    sample_mask = engine.read_ciphertext(session_dir / "sample_mask.ct")

    params_dir = session_dir / "params"
    params = {
        "alpha": [engine.read_ciphertext(params_dir / f"alpha_{i}.ct") for i in range(n_internal)],
        "threshold": [
            [engine.read_ciphertext(params_dir / f"threshold_{i}_{j}.ct") for j in range(n_features)]
            for i in range(n_internal)
        ],
        "leaf_logits": [engine.read_ciphertext(params_dir / f"leaf_{l}.ct") for l in range(n_leaves)],
    }

    new_params = forward_backward_update_N(
        ctx, dataset, params, sample_mask, n_features, config["n_classes"], depth, lr=config["lr"]
    )

    for i in range(n_internal):
        engine.write_ciphertext(new_params["alpha"][i], params_dir / f"alpha_{i}.ct")
        for j in range(n_features):
            engine.write_ciphertext(new_params["threshold"][i][j], params_dir / f"threshold_{i}_{j}.ct")
    for l in range(n_leaves):
        engine.write_ciphertext(new_params["leaf_logits"][l], params_dir / f"leaf_{l}.ct")


if __name__ == "__main__":
    main()
