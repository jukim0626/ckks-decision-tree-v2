"""train.py 오케스트레이터들이 GPU 메모리 상태를 확인/대기하는 데 공통으로 쓰는 유틸.

Soft Decision Tree 알고리즘과 무관한 순수 프로세스/GPU 관리 로직 - baseline/packed/
local_loss/opt의 train.py에 바이트 단위로 동일하게 복붙돼 있던 걸 여기로 뽑았다
(2026-09-21 1차 리팩터, 수치 동작은 전혀 안 바꿈)."""

from __future__ import annotations

import ctypes
import subprocess
import sys
import threading
import time

# 2026-10-26: unattended-upgrade로 커널 모듈/유저 lib 버전이 어긋나면 nvidia-smi(NVML)만 깨지고
# CUDA는 정상이다. 그때는 CUDA driver API(cuMemGetInfo)로 대신 잰다.
_CUDA_MEM_SNIPPET = """
import ctypes, sys
c = ctypes.CDLL("libcuda.so.1")
dev, ctx, free, total = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_size_t(), ctypes.c_size_t()
if (c.cuInit(0) or c.cuDeviceGet(ctypes.byref(dev), 0) or c.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
        or c.cuCtxSetCurrent(ctx) or c.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))):
    sys.exit(1)
print((total.value - free.value) // 2**20)
"""


def _nvml_memory_used_mib() -> int | None:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    return int(out[0]) if out and out[0].isdigit() else None


def _cuda_memory_used_mib() -> int:
    """별도 프로세스에서 잰다 - 오케스트레이터 프로세스에 CUDA context가 남으면 그만큼 계속 점유하므로.
    측정 프로세스 자신의 context(수백 MiB)도 포함된 값이라 절대 threshold 비교엔 못 쓴다."""
    out = subprocess.run([sys.executable, "-c", _CUDA_MEM_SNIPPET], capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 0


def _cuda_memory_used_mib_inprocess() -> int:
    """이미 CUDA를 쓰는 worker 프로세스 안에서 쓰는 버전(primary context를 공유하므로 추가 점유 없음)."""
    c = ctypes.CDLL("libcuda.so.1")
    dev, ctx, free, total = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_size_t(), ctypes.c_size_t()
    if (c.cuInit(0) or c.cuDeviceGet(ctypes.byref(dev), 0) or c.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
            or c.cuCtxSetCurrent(ctx) or c.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))):
        return 0
    return (total.value - free.value) // 2**20


def gpu_memory_used_mib() -> int:
    used = _nvml_memory_used_mib()
    return used if used is not None else _cuda_memory_used_mib_inprocess()


def wait_for_gpu_settle(threshold_mib: int = 500, timeout_s: float = 60.0, poll_s: float = 1.0) -> None:
    """직전 worker 프로세스가 죽은 직후에도 CUDA driver의 GPU 메모리 회수가 살짝 지연될 수
    있어서, 다음 프로세스를 띄우기 전에 실제로 메모리가 threshold 밑으로 떨어질 때까지 짧게
    폴링한다 (baseline/train.py 2026-08-26 실측에서 발견한 레이스 컨디션 우회).
    NVML이 깨졌을 때는 측정값에 측정 프로세스 자신의 context가 섞여 threshold를 못 쓰므로,
    사용량이 더 안 줄어들 때(연속 두 번 64MiB 이내)까지 기다린다."""
    t0 = time.time()
    prev = None
    while time.time() - t0 < timeout_s:
        used = _nvml_memory_used_mib()
        if used is not None:
            if used < threshold_mib:
                return
        else:
            used = _cuda_memory_used_mib()
            if prev is not None and abs(prev - used) <= 64:
                return
            prev = used
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
