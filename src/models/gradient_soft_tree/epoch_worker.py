"""depth=1 gradient soft tree의 한 epoch(forward+backward+SGD update)을 별도 프로세스로
실행한다. desilofhe는 key가 살아있는 프로세스에서 만든 ciphertext의 GPU 메모리 풀을
계속 붙잡고 있어서(EXPERIMENT_LOG.md 2026-08-11/12, closed_form_mgi/node_worker.py와
동일 원인) 여러 epoch을 한 프로세스에서 돌리면 GPU 메모리가 계속 쌓인다 - 실측으로
`depth1_ckks.py`의 단일 프로세스 버전이 epoch 3에서 CUDA OOM으로 죽었다. node_worker.py/
depth_worker.py와 같은 원리로, epoch 하나마다 새 프로세스를 띄워 끝나면 OS가 GPU 메모리를
강제로 회수하게 만든다.

session_dir/params/*.ct를 읽어서 한 epoch 갱신한 뒤 같은 자리에 덮어쓴다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data.serialization import load_context, read_dataset  # noqa: E402
from core.ckks_engine import create_bootstrap_engine  # noqa: E402
from models.gradient_soft_tree.depth1_ckks import forward_backward_update  # noqa: E402


def main() -> None:
    session_dir = Path(sys.argv[1])
    config = json.loads((session_dir / "config.json").read_text())
    n_features = config["n_features"]

    engine = create_bootstrap_engine(mode=config["mode"], device_id=config["device_id"])
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
        "alpha": engine.read_ciphertext(params_dir / "alpha.ct"),
        "threshold": [engine.read_ciphertext(params_dir / f"threshold_{j}.ct") for j in range(n_features)],
        "leaf_L": engine.read_ciphertext(params_dir / "leaf_L.ct"),
        "leaf_R": engine.read_ciphertext(params_dir / "leaf_R.ct"),
    }

    new_params = forward_backward_update(
        ctx, dataset, params, sample_mask, n_features, config["n_classes"], lr=config["lr"]
    )

    engine.write_ciphertext(new_params["alpha"], params_dir / "alpha.ct")
    for j, t in enumerate(new_params["threshold"]):
        engine.write_ciphertext(t, params_dir / f"threshold_{j}.ct")
    engine.write_ciphertext(new_params["leaf_L"], params_dir / "leaf_L.ct")
    engine.write_ciphertext(new_params["leaf_R"], params_dir / "leaf_R.ct")


if __name__ == "__main__":
    main()
