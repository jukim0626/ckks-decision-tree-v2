"""train_depthN_ckks.py(axis-aligned)와 완전히 같은 오케스트레이터 - worker 모듈 경로만
oblique.*로 바꿈. GPU 메모리 관련 주의사항(오케스트레이터 자신은 GPU를 안 잡음, epoch 사이
GPU 메모리 회수 대기)도 전부 동일하게 적용된다.

사용법: python -m experiments.gradient_soft_tree.oblique.train_depthN_ckks iris 2 30 1.0 0 17
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


def train(dataset_name: str, depth: int, n_epochs: int, lr: float, seed: int, level_preset: int | None = 17, max_train: int | None = None) -> dict:
    session_dir = Path(tempfile.mkdtemp(prefix=f"oblique_gradient_soft_tree_depth{depth}_"))

    print(
        f"[setup] dataset={dataset_name} depth={depth} epochs={n_epochs} lr={lr} seed={seed} "
        f"level_preset={level_preset} max_train={max_train}",
        flush=True,
    )
    _wait_for_gpu_settle()
    _run(
        "experiments.gradient_soft_tree.oblique.setup_worker_N",
        str(session_dir), dataset_name, str(depth), str(seed), str(lr),
        str(level_preset) if level_preset is not None else "none",
        str(max_train) if max_train is not None else "none",
    )

    for epoch in range(1, n_epochs + 1):
        _wait_for_gpu_settle()
        t0 = time.time()
        _run("experiments.gradient_soft_tree.oblique.epoch_worker_N", str(session_dir))
        elapsed = time.time() - t0

        # 2026-09-07 밤 실험 이후 추가: 실제 CKKS 파라미터의 z 범위를 매 epoch 관찰만 함
        # (자동 개입 없음 - 언제/어떻게 plaintext trajectory에서 벗어나 위험구간에 들어가는지
        # 데이터를 먼저 모으는 진단 단계, check_worker_N.py 참고)
        _wait_for_gpu_settle()
        check_stdout = _run("experiments.gradient_soft_tree.oblique.check_worker_N", str(session_dir), str(epoch))
        check = json.loads(check_stdout.strip().splitlines()[-1])
        danger = " ⚠️ 안전구간(2.0) 이탈" if check["max_z"] > 2.0 else ""
        print(
            f"[oblique {dataset_name} depth={depth}] epoch {epoch}/{n_epochs} done | {elapsed:.1f}s | "
            f"max|z|={check['max_z']:.3f} max|w|={check['max_w_abs']:.3f} max|b|={check['max_b_abs']:.3f}{danger}",
            flush=True,
        )

    _wait_for_gpu_settle()
    stdout = _run("experiments.gradient_soft_tree.oblique.finalize_worker_N", str(session_dir), str(n_epochs))
    result = json.loads(stdout.strip().splitlines()[-1])
    print(
        f"[oblique {dataset_name} depth={depth}] max abs diff vs plaintext = {result['max_err']:.5f} | "
        f"train_acc={result['train_acc']:.4f} test_acc={result['test_acc']:.4f}",
        flush=True,
    )
    print(f"[oblique {dataset_name} depth={depth}] session_dir={session_dir}", flush=True)
    return result


def main() -> None:
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "iris"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    level_preset_arg = sys.argv[6] if len(sys.argv) > 6 else "17"
    level_preset = None if level_preset_arg == "none" else int(level_preset_arg)
    train(dataset_name, depth, n_epochs, lr, seed, level_preset=level_preset)


if __name__ == "__main__":
    main()
