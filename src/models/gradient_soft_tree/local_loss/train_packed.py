"""train.py(local_loss)와 완전히 같은 오케스트레이션이지만, epoch마다
epoch_worker_packed(forward_backward_update_N_packed, feature-axis SIMD packing)를 쓴다.
setup/finalize는 그대로 재사용(params[] 포맷이 packing과 무관).

사용법: python -m models.gradient_soft_tree.local_loss.train_packed wine 3 30 6.0 0 17
        (dataset, depth, epochs, lr, seed, level_preset, [max_train])
breast_cancer(30 feature)는 기본 train set(539개)로는 feature-axis 레이아웃이 slot_count를
넘쳐서(30*2048=61440 > 32768) --max-train 512 이하가 필요하다(30*1024=30720으로 들어감)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.runtime.gpu import wait_for_gpu_settle  # noqa: E402
from core.runtime.session import make_session_dir, run_worker_with_oom_retry  # noqa: E402
from core.runtime.subprocess_worker import run_worker_module  # noqa: E402

_MAX_OOM_RETRIES = 3


def train(
    dataset_name: str,
    depth: int,
    n_epochs: int,
    lr: float,
    seed: int,
    level_preset: int | None = 17,
    max_train: int | None = None,
) -> dict:
    session_dir = make_session_dir(f"local_loss_packed_depth{depth}_")

    print(
        f"[setup] dataset={dataset_name} depth={depth} epochs={n_epochs} lr={lr} seed={seed} "
        f"level_preset={level_preset} max_train={max_train}",
        flush=True,
    )
    wait_for_gpu_settle()
    run_worker_module(
        "models.gradient_soft_tree.local_loss.setup_worker",
        str(session_dir), dataset_name, str(depth), str(seed), str(lr),
        str(level_preset) if level_preset is not None else "none",
        str(max_train) if max_train is not None else "none",
    )

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        # packed/train.py와 동일한 이유(breast_cancer depth=4에서 2026-09-13 실측 확인:
        # depth=4는 노드가 15개라 ciphertext가 훨씬 많아져서 GPU 메모리 회수가 지연될 때
        # OOM이 남) - epoch_worker_packed는 params를 다 계산한 뒤에만 디스크에 쓰므로
        # OOM으로 죽어도 재시도가 안전하다.
        run_worker_with_oom_retry(
            lambda: run_worker_module("models.gradient_soft_tree.local_loss.epoch_worker_packed", str(session_dir)),
            epoch=epoch, log_prefix=f"[local_loss_packed {dataset_name} depth={depth}]", max_retries=_MAX_OOM_RETRIES,
        )
        elapsed = time.time() - t0
        print(f"[local_loss_packed {dataset_name} depth={depth}] epoch {epoch}/{n_epochs} done | {elapsed:.1f}s", flush=True)

    wait_for_gpu_settle()
    stdout = run_worker_module("models.gradient_soft_tree.local_loss.finalize_worker", str(session_dir), str(n_epochs))
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
