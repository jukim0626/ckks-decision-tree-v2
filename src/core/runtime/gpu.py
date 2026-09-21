"""train.py 오케스트레이터들이 GPU 메모리 상태를 확인/대기하는 데 공통으로 쓰는 유틸.

Soft Decision Tree 알고리즘과 무관한 순수 프로세스/GPU 관리 로직 - baseline/packed/
local_loss/opt의 train.py에 바이트 단위로 동일하게 복붙돼 있던 걸 여기로 뽑았다
(2026-09-21 1차 리팩터, 수치 동작은 전혀 안 바꿈)."""

from __future__ import annotations

import subprocess
import threading
import time


def gpu_memory_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    return int(out[0]) if out else 0


def wait_for_gpu_settle(threshold_mib: int = 500, timeout_s: float = 60.0, poll_s: float = 1.0) -> None:
    """직전 worker 프로세스가 죽은 직후에도 CUDA driver의 GPU 메모리 회수가 살짝 지연될 수
    있어서, 다음 프로세스를 띄우기 전에 실제로 메모리가 threshold 밑으로 떨어질 때까지 짧게
    폴링한다 (baseline/train.py 2026-08-26 실측에서 발견한 레이스 컨디션 우회)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if gpu_memory_used_mib() < threshold_mib:
            return
        time.sleep(poll_s)


class GpuPeakWatcher:
    """백그라운드 스레드로 GPU 메모리 사용량 peak을 폴링(opt/train_opt.py에서 이동,
    로직 변경 없음). with 블록 동안의 최대 사용량이 self.peak에 남는다."""

    def __init__(self, poll_s: float = 1.0):
        self.poll_s = poll_s
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, gpu_memory_used_mib())
            time.sleep(self.poll_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
