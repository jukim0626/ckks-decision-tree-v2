"""train.py(local_loss)와 완전히 같은 오케스트레이션이지만, epoch마다
epoch_worker_packed(forward_backward_update_N_packed, feature-axis SIMD packing)를 쓴다.
setup/finalize는 그대로 재사용(params[] 포맷이 packing과 무관).

사용법: python -m models.gradient_soft_tree.local_loss.train_packed wine 3 30 6.0 0 17
        (dataset, depth, epochs, lr, seed, level_preset, [max_train])
breast_cancer(30 feature)는 기본 train set(539개)로는 feature-axis 레이아웃이 slot_count를
넘쳐서(30*2048=61440 > 32768) --max-train 512 이하가 필요하다(30*1024=30720으로 들어감)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO_ROOT = Path(__file__).resolve().parents[3]
_MAX_OOM_RETRIES = 3


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


def train(
    dataset_name: str,
    depth: int,
    n_epochs: int,
    lr: float,
    seed: int,
    level_preset: int | None = 17,
    max_train: int | None = None,
) -> dict:
    session_dir = Path(tempfile.mkdtemp(prefix=f"local_loss_packed_depth{depth}_"))

    print(
        f"[setup] dataset={dataset_name} depth={depth} epochs={n_epochs} lr={lr} seed={seed} "
        f"level_preset={level_preset} max_train={max_train}",
        flush=True,
    )
    _wait_for_gpu_settle()
    _run(
        "models.gradient_soft_tree.local_loss.setup_worker",
        str(session_dir), dataset_name, str(depth), str(seed), str(lr),
        str(level_preset) if level_preset is not None else "none",
        str(max_train) if max_train is not None else "none",
    )

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        for attempt in range(1, _MAX_OOM_RETRIES + 2):
            _wait_for_gpu_settle()
            try:
                _run("models.gradient_soft_tree.local_loss.epoch_worker_packed", str(session_dir))
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc) or attempt > _MAX_OOM_RETRIES:
                    raise
                # epoch_worker_packed는 params를 다 계산한 뒤에만 디스크에 쓰므로(중간에 안
                # 씀), OOM으로 죽어도 params는 직전 epoch 상태 그대로 남아있어 재시도가
                # 안전하다 - packed/train.py와 동일한 이유(breast_cancer depth=4에서
                # 2026-09-13 실측 확인: depth=4는 노드가 15개라 ciphertext가 훨씬 많아져서
                # GPU 메모리 회수가 지연될 때 OOM이 남).
                wait_s = 30.0 * attempt
                print(
                    f"[local_loss_packed {dataset_name} depth={depth}] epoch {epoch} OOM (attempt {attempt}/"
                    f"{_MAX_OOM_RETRIES}) - {wait_s:.0f}초 대기 후 재시도",
                    flush=True,
                )
                time.sleep(wait_s)
        elapsed = time.time() - t0
        print(f"[local_loss_packed {dataset_name} depth={depth}] epoch {epoch}/{n_epochs} done | {elapsed:.1f}s", flush=True)

    _wait_for_gpu_settle()
    stdout = _run("models.gradient_soft_tree.local_loss.finalize_worker", str(session_dir), str(n_epochs))
    result = json.loads(stdout.strip().splitlines()[-1])
    print(
        f"[local_loss_packed {dataset_name} depth={depth}] max abs diff vs plaintext = {result['max_err']:.5f} | "
        f"train_acc={result['train_acc']:.4f} test_acc={result['test_acc']:.4f}",
        flush=True,
    )
    print(f"[local_loss_packed {dataset_name} depth={depth}] session_dir={session_dir}", flush=True)
    return result


def main() -> None:
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "iris"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    level_preset_arg = sys.argv[6] if len(sys.argv) > 6 else "17"
    level_preset = None if level_preset_arg == "none" else int(level_preset_arg)
    max_train_arg = sys.argv[7] if len(sys.argv) > 7 else "none"
    max_train = None if max_train_arg == "none" else int(max_train_arg)
    train(dataset_name, depth, n_epochs, lr, seed, level_preset=level_preset, max_train=max_train)


if __name__ == "__main__":
    main()
