"""train_depthN_ckks.py와 같은 오케스트레이션 패턴(setup/epoch/finalize를 전부 별도
프로세스로 격리)이지만, forward_backward_update_N_opt를 쓰는 epoch_worker_opt.py를 호출하고
TreeConfig(experiments_registry.py의 preset)를 session_dir/experiment_config.json에 써서
넘긴다. setup_worker_N.py/finalize는 opt 버전을 쓴다(finalize는 config별 predict/reference가
필요해서 opt 전용, setup은 baseline과 encoding이 완전히 같아서 그대로 재사용).

사용법: python -m experiments.gradient_soft_tree.opt.train_opt iris 3 <preset> <epochs> <lr> <seed> <level_preset> [detailed]
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.gradient_soft_tree.opt.experiments_registry import get_preset  # noqa: E402

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
    result = subprocess.run([sys.executable, "-m", module, *args], capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"{module} 실패:\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    return result.stdout


class _GpuPeakWatcher:
    def __init__(self, poll_s: float = 1.0):
        self.poll_s = poll_s
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, _gpu_memory_used_mib())
            time.sleep(self.poll_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


def train(
    dataset_name: str, depth: int, preset_name: str, n_epochs: int, lr: float, seed: int,
    level_preset: int | None = 17, detailed: bool = False,
) -> dict:
    tree_config = get_preset(preset_name)
    session_dir = Path(tempfile.mkdtemp(prefix=f"opt_{preset_name}_depth{depth}_"))

    print(
        f"[setup] preset={preset_name} dataset={dataset_name} depth={depth} epochs={n_epochs} lr={lr} "
        f"seed={seed} level_preset={level_preset}",
        flush=True,
    )
    _wait_for_gpu_settle()
    _run(
        "experiments.gradient_soft_tree.setup_worker_N",
        str(session_dir), dataset_name, str(depth), str(seed), str(lr),
        str(level_preset) if level_preset is not None else "none",
    )
    (session_dir / "experiment_config.json").write_text(json.dumps(dataclasses.asdict(tree_config)))

    epoch_results = []
    peak_mib = 0
    for epoch in range(1, n_epochs + 1):
        _wait_for_gpu_settle()
        t0 = time.time()
        with _GpuPeakWatcher() as watcher:
            args = [str(session_dir)] + (["detailed"] if detailed else [])
            stdout = _run("experiments.gradient_soft_tree.opt.epoch_worker_opt", *args)
        peak_mib = max(peak_mib, watcher.peak)
        elapsed = time.time() - t0
        lines = stdout.strip().splitlines()
        print("\n".join(lines[:-1]), flush=True)  # profiler summary/trace (마지막 json 줄 제외)
        epoch_json = json.loads(lines[-1])
        epoch_results.append(epoch_json)
        print(
            f"[{preset_name} {dataset_name} depth={depth}] epoch {epoch}/{n_epochs} done | wall={elapsed:.1f}s "
            f"bootstrap={epoch_json['bootstrap_count']} bootstrap_s={epoch_json['bootstrap_time_s']:.1f} "
            f"peak_gpu_mib={watcher.peak}",
            flush=True,
        )

    _wait_for_gpu_settle()
    stdout = _run("experiments.gradient_soft_tree.opt.finalize_worker_opt", str(session_dir), str(n_epochs))
    result = json.loads(stdout.strip().splitlines()[-1])
    result["preset"] = preset_name
    result["epoch_results"] = epoch_results
    result["peak_gpu_mib"] = peak_mib
    result["session_dir"] = str(session_dir)
    print(
        f"[{preset_name} {dataset_name} depth={depth}] max abs diff vs plaintext = {result['max_err']:.5f} | "
        f"train_acc={result['train_acc']:.4f} test_acc={result['test_acc']:.4f} peak_gpu_mib={peak_mib}",
        flush=True,
    )
    print(f"[{preset_name} {dataset_name} depth={depth}] session_dir={session_dir}", flush=True)
    (session_dir / "result.json").write_text(json.dumps(result, indent=2))
    _append_manifest(preset_name, dataset_name, depth, n_epochs, lr, seed, level_preset, session_dir)
    return result


def _append_manifest(preset_name, dataset_name, depth, n_epochs, lr, seed, level_preset, session_dir: Path) -> None:
    manifest = REPO_ROOT / "results" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "preset": preset_name, "dataset": dataset_name, "depth": depth, "n_epochs": n_epochs,
        "lr": lr, "seed": seed, "level_preset": level_preset, "session_dir": str(session_dir),
        "recorded_at": time.time(),
    }
    with open(manifest, "a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "iris"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    preset_name = sys.argv[3] if len(sys.argv) > 3 else "baseline"
    n_epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    lr = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    level_preset_arg = sys.argv[7] if len(sys.argv) > 7 else "17"
    level_preset = None if level_preset_arg == "none" else int(level_preset_arg)
    detailed = len(sys.argv) > 8 and sys.argv[8] == "detailed"
    train(dataset_name, depth, preset_name, n_epochs, lr, seed, level_preset=level_preset, detailed=detailed)


if __name__ == "__main__":
    main()
