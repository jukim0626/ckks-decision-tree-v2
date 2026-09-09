"""depthN_ckks의 forward_backward_update_N을 epoch마다 별도 프로세스(epoch_worker_N.py)로
실행하는 오케스트레이터. setup(setup_worker_N.py)과 최종 평가(finalize_worker_N.py)도 각각
별도 프로세스로 분리했다 - **오케스트레이터 자신은 GPU를 절대 만지지 않는다.**

**2026-08-26 depth=3 실패로 발견한 교훈**: 처음 버전은 오케스트레이터가 setup(dataset/초기
파라미터 암호화)을 자기 프로세스에서 직접 했는데, 그러면 오케스트레이터가 epoch_worker_N을
띄우는 동안 *자기 자신의* GPU context/키도 계속 살아있어서 depth=3(초기 ciphertext 43개)에서
부모+자식 합산 메모리가 24GB를 넘어 epoch 1부터 OOM이 났다(depth=1은 파라미터가 8개뿐이라
우연히 버팀). setup/finalize를 전부 별도 프로세스로 빼서 오케스트레이터 프로세스 자체는 GPU
메모리를 0으로 유지한다.

사용법: python -m models.gradient_soft_tree.baseline.train_depthN_ckks iris 3 35 2.0 0
        (dataset, depth, epochs, lr, seed)
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
    """epoch_worker_N 프로세스가 죽은 직후에도 CUDA driver의 GPU 메모리 회수가 살짝 지연될
    수 있다 (2026-08-26 실측: depth=3 epoch 1이 완전히 끝나고 프로세스가 종료됐는데도 바로
    다음 epoch_worker_N을 띄우면 OOM - depth=3의 단일 epoch peak(~19~20GB)이 24GB 한도에
    너무 바짝 붙어있어서, 이 찰나의 회수 지연과 겹치면 다음 프로세스가 시작부터 메모리가
    없는 것으로 추정). 다음 프로세스를 띄우기 전에 GPU 메모리가 threshold 밑으로 실제
    떨어질 때까지 짧게 폴링해서 이 레이스를 없앤다."""
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
    session_dir = Path(tempfile.mkdtemp(prefix=f"gradient_soft_tree_depth{depth}_"))

    print(
        f"[setup] dataset={dataset_name} depth={depth} epochs={n_epochs} lr={lr} seed={seed} "
        f"level_preset={level_preset}",
        flush=True,
    )
    _wait_for_gpu_settle()
    _run(
        "models.gradient_soft_tree.baseline.setup_worker_N",
        str(session_dir), dataset_name, str(depth), str(seed), str(lr),
        str(level_preset) if level_preset is not None else "none",
    )

    for epoch in range(1, n_epochs + 1):
        _wait_for_gpu_settle()
        t0 = time.time()
        _run("models.gradient_soft_tree.baseline.epoch_worker_N", str(session_dir))
        elapsed = time.time() - t0
        print(f"[{dataset_name} depth={depth}] epoch {epoch}/{n_epochs} done | {elapsed:.1f}s", flush=True)

    _wait_for_gpu_settle()
    stdout = _run("models.gradient_soft_tree.baseline.finalize_worker_N", str(session_dir), str(n_epochs))
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
