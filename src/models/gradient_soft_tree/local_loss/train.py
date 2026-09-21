"""local_loss forward_backward_update_N을 epoch마다 별도 프로세스(epoch_worker.py)로
실행하는 오케스트레이터. baseline/train.py와 완전히 같은 구조 - setup/epoch/finalize를
각각 별도 프로세스로 분리해서 오케스트레이터 자신은 GPU를 절대 만지지 않는다.

사용법: python -m models.gradient_soft_tree.local_loss.train iris 3 1 2.0 0 14
        (dataset, depth, epochs, lr, seed, level_preset)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.runtime.gpu import wait_for_gpu_settle  # noqa: E402
from core.runtime.session import make_session_dir  # noqa: E402
from core.runtime.subprocess_worker import run_worker_module  # noqa: E402


def train(dataset_name: str, depth: int, n_epochs: int, lr: float, seed: int, level_preset: int | None = 17) -> dict:
    session_dir = make_session_dir(f"local_loss_depth{depth}_")

    print(
        f"[setup] dataset={dataset_name} depth={depth} epochs={n_epochs} lr={lr} seed={seed} "
        f"level_preset={level_preset}",
        flush=True,
    )
    wait_for_gpu_settle()
    run_worker_module(
        "models.gradient_soft_tree.local_loss.setup_worker",
        str(session_dir), dataset_name, str(depth), str(seed), str(lr),
        str(level_preset) if level_preset is not None else "none",
    )

    for epoch in range(1, n_epochs + 1):
        wait_for_gpu_settle()
        t0 = time.time()
        run_worker_module("models.gradient_soft_tree.local_loss.epoch_worker", str(session_dir))
        elapsed = time.time() - t0
        print(f"[{dataset_name} depth={depth}] epoch {epoch}/{n_epochs} done | {elapsed:.1f}s", flush=True)

    wait_for_gpu_settle()
    stdout = run_worker_module("models.gradient_soft_tree.local_loss.finalize_worker", str(session_dir), str(n_epochs))
    result = json.loads(stdout.strip().splitlines()[-1])
    print(
        f"[{dataset_name} depth={depth}] max abs diff vs plaintext = {result['max_err']:.5f} | "
        f"train_acc={result['train_acc']:.4f} test_acc={result['test_acc']:.4f}",
        flush=True,
    )
    print(f"[{dataset_name} depth={depth}] session_dir={session_dir}", flush=True)
    return result


def main() -> None:
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "iris"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 35
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    level_preset_arg = sys.argv[6] if len(sys.argv) > 6 else "17"
    level_preset = None if level_preset_arg == "none" else int(level_preset_arg)
    train(dataset_name, depth, n_epochs, lr, seed, level_preset=level_preset)


if __name__ == "__main__":
    main()
