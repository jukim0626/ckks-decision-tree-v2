"""학습 session_dir 생성 + OOM 재시도 루프. baseline/packed/local_loss/opt의 train.py가
공통으로 쓰던 패턴을 뽑았다(2026-09-21 1차 리팩터, 수치 동작은 전혀 안 바꿈 - 재시도
횟수/대기 시간/"out of memory" 판별 조건 모두 packed/train.py, local_loss/train_packed.py
원본과 동일하게 유지)."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Callable

from core.runtime.gpu import wait_for_gpu_settle


def make_session_dir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def run_worker_with_oom_retry(
    run_fn: Callable[[], str],
    *,
    epoch: int,
    log_prefix: str,
    max_retries: int = 3,
    base_wait_s: float = 30.0,
) -> str:
    """`run_fn`(보통 `run_worker_module`을 부르는 람다)을 실행하다 "out of memory" 에러가
    나면 최대 `max_retries`번 재시도한다 - epoch_worker_packed류는 params를 다 계산한
    뒤에만 디스크에 쓰므로(중간에 안 씀) OOM으로 죽어도 params가 직전 epoch 상태 그대로
    남아있어 재시도가 안전하다 (packed/train.py, local_loss/train_packed.py 공통 전제)."""
    for attempt in range(1, max_retries + 2):
        wait_for_gpu_settle()
        try:
            return run_fn()
        except RuntimeError as exc:
            if "out of memory" not in str(exc) or attempt > max_retries:
                raise
            wait_s = base_wait_s * attempt
            print(
                f"{log_prefix} epoch {epoch} OOM (attempt {attempt}/{max_retries}) - {wait_s:.0f}초 대기 후 재시도",
                flush=True,
            )
            time.sleep(wait_s)
    raise AssertionError("unreachable")
