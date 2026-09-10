"""local_loss forward_backward_update_N을 epoch마다 별도 프로세스(epoch_worker.py)로
실행하는 오케스트레이터. baseline/train.py와 완전히 같은 구조 - setup/epoch/finalize를
각각 별도 프로세스로 분리해서 오케스트레이터 자신은 GPU를 절대 만지지 않는다.

사용법: python -m models.gradient_soft_tree.local_loss.train iris 3 1 2.0 0 14
        (dataset, depth, epochs, lr, seed, level_preset)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO_ROOT = Path(__file__).resolve().parents[3]


def _gpu_memory_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    return int(out[0]) if out else 0


def _wait_for_gpu_settle(threshold_mib: int = 500, timeout_s: float = 60.0, poll_s: float = 1.0) -> None:
    """epoch_worker 프로세스가 죽은 직후에도 GPU 메모리 회수가 살짝 지연될 수 있어서, 다음
    프로세스를 띄우기 전에 실제로 메모리가 threshold 밑으로 떨어질 때까지 짧게 폴링한다
    (baseline/train.py와 동일한 이유)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if _gpu_memory_used_mib() < threshold_mib:
            return
        time.sleep(poll_s)


def _run(module: str, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"{module} 실패:\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    return result.stdout


def train(dataset_name: str, depth: int, n_epochs: int, lr: float, seed: int, level_preset: int | None = 17) -> dict:
    session_dir = Path(tempfile.mkdtemp(prefix=f"local_loss_depth{depth}_"))

    print(
        f"[setup] dataset={dataset_name} depth={depth} epochs={n_epochs} lr={lr} seed={seed} "
        f"level_preset={level_preset}",
        flush=True,
    )
    _wait_for_gpu_settle()
    _run(
        "models.gradient_soft_tree.local_loss.setup_worker",
        str(session_dir), dataset_name, str(depth), str(seed), str(lr),
        str(level_preset) if level_preset is not None else "none",
    )

    for epoch in range(1, n_epochs + 1):
        _wait_for_gpu_settle()
        t0 = time.time()
        _run("models.gradient_soft_tree.local_loss.epoch_worker", str(session_dir))
        elapsed = time.time() - t0
        print(f"[{dataset_name} depth={depth}] epoch {epoch}/{n_epochs} done | {elapsed:.1f}s", flush=True)

    _wait_for_gpu_settle()
    stdout = _run("models.gradient_soft_tree.local_loss.finalize_worker", str(session_dir), str(n_epochs))
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
