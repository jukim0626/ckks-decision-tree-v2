"""`packed/train.py`와 완전히 같은 오케스트레이션이지만, epoch마다
`epoch_worker_packed_wide`(feature를 여러 ciphertext 그룹으로 나누는 버전)를 쓴다 -
`n_features*block_size > slot_count`라 단일 ciphertext 경로(`packed/train.py`)로는
`assert_layout_fits`에서 바로 실패하는 넓은 데이터셋(예: soybean, 35 feature) 전용.
setup/finalize는 baseline 그대로 재사용 - params[] 포맷이 단일 ciphertext 버전과 완전히
동일하므로(threshold도 baseline과 같은 feature별 개별 ciphertext 리스트) 그대로 호환된다.

사용법: python -m models.gradient_soft_tree.packed.train_wide soybean 2 10 2.0 0 17
"""

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
    test_size: float | int = 0.2,
) -> dict:
    session_dir = make_session_dir(f"packed_wide_gradient_soft_tree_depth{depth}_")

    print(
        f"[setup] dataset={dataset_name} depth={depth} epochs={n_epochs} lr={lr} seed={seed} "
        f"level_preset={level_preset} max_train={max_train} test_size={test_size}",
        flush=True,
    )
    wait_for_gpu_settle()
    run_worker_module(
        "models.gradient_soft_tree.baseline.setup_worker",
        str(session_dir), dataset_name, str(depth), str(seed), str(lr),
        str(level_preset) if level_preset is not None else "none",
        str(max_train) if max_train is not None else "none",
        str(test_size),
    )

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        run_worker_with_oom_retry(
            lambda: run_worker_module("models.gradient_soft_tree.packed.epoch_worker_packed_wide", str(session_dir)),
            epoch=epoch, log_prefix=f"[packed_wide {dataset_name} depth={depth}]", max_retries=_MAX_OOM_RETRIES,
        )
        elapsed = time.time() - t0
        print(f"[packed_wide {dataset_name} depth={depth}] iteration {epoch}/{n_epochs} (full-batch GD step) done | {elapsed:.1f}s", flush=True)

    wait_for_gpu_settle()
    stdout = run_worker_module("models.gradient_soft_tree.baseline.finalize_worker", str(session_dir), str(n_epochs))
    result = json.loads(stdout.strip().splitlines()[-1])
    print(
        f"[packed_wide {dataset_name} depth={depth}] [true-fn] max abs diff vs plaintext = {result['max_err']:.5f} | "
        f"train_acc={result['train_acc']:.4f} test_acc={result['test_acc']:.4f}",
        flush=True,
    )
    print(
        f"[packed_wide {dataset_name} depth={depth}] [poly]    max abs diff vs plaintext = {result['max_err_poly']:.5f} | "
        f"train_acc={result['train_acc_poly']:.4f} test_acc={result['test_acc_poly']:.4f}",
        flush=True,
    )
    for split in ("train", "test"):
        d = result[f"domain_{split}"]
        flag = "  <-- 다항식 구간 이탈!" if d["n_samples_gate_out_of_range"] or d["softmax_out_of_range"] else ""
        print(
            f"[packed_wide {dataset_name} depth={depth}] domain({split}): max|x-t|={d['max_abs_gate_input']:.3f}(<=2) "
            f"out_of_range_samples={d['n_samples_gate_out_of_range']} "
            f"max|alpha|={d['max_abs_alpha']:.3f} max|leaf|={d['max_abs_leaf_logit']:.3f}(<=2.5){flag}",
            flush=True,
        )
    print(f"[packed_wide {dataset_name} depth={depth}] session_dir={session_dir}", flush=True)
    return result


def main() -> None:
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "soybean"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    level_preset_arg = sys.argv[6] if len(sys.argv) > 6 else "17"
    level_preset = None if level_preset_arg == "none" else int(level_preset_arg)
    max_train_arg = sys.argv[7] if len(sys.argv) > 7 else "none"
    max_train = None if max_train_arg == "none" else int(max_train_arg)
    test_size_arg = sys.argv[8] if len(sys.argv) > 8 else "0.2"
    test_size = float(test_size_arg) if "." in test_size_arg else int(test_size_arg)
    train(dataset_name, depth, n_epochs, lr, seed, level_preset=level_preset, max_train=max_train, test_size=test_size)


if __name__ == "__main__":
    main()
