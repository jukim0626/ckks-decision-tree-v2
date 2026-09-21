"""`python -m <module>`로 worker 프로세스를 띄우고 결과를 회수하는 공통 wrapper.

오케스트레이터(train.py) 자신이 GPU를 절대 안 만지고 모든 GPU 작업을 별도 프로세스로
격리하는 패턴(desilofhe가 프로세스 안에서 GPU 메모리를 절대 안 돌려주는 문제 우회)이
baseline/packed/local_loss/opt 전부에 공통이라, 그 실행부(`_run`)만 뽑았다
(2026-09-21 1차 리팩터, 수치 동작은 전혀 안 바꿈). 이름을 파일명으로도 `subprocess`가
아니라 `subprocess_worker`로 둔 이유: 표준 라이브러리 `subprocess` 모듈과 이름이 겹치면
이 모듈 내부의 `import subprocess`가 자기 자신을 가리키는 사고가 나기 쉽다."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# src/core/runtime/subprocess_worker.py 기준 parents[2] == src/ (REPO_ROOT).
REPO_ROOT = Path(__file__).resolve().parents[2]


def run_worker_module(module: str, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"{module} 실패:\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    return result.stdout
