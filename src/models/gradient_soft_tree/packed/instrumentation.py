"""packed 학습(joint training, vanilla GD)의 forward/backward/parameter-update를
계측하기 위한 도구. `opt/engine_counter.py`(CountingEngineProxy)와
`opt/profiler.py`(BootstrapProfiler)의 아이디어를 재사용하되, 이번 최적화 작업이
요구하는 지표(ct-ct vs ct-pt multiply 구분, merge_bootstrap 카운트, refresh된
ciphertext 개수, phase별 GPU-sync 기반 wall time)에 맞게 확장했다. baseline/opt의
기존 계측 코드는 건드리지 않는다(별도 모듈).

**GPU 비동기 실행에 대한 조사 결과 (2026-09-21)**: 설치된 `desilofhe-cu130==1.14.1`의
`Engine`에 인자 없는 `sync() -> None` 메서드가 있다. `Engine(mode="cpu")`에서
`.sync()`를 실제로 호출해보니 `RuntimeError: Engine is not in Async GPU mode`가
났다 - **CPU 모드는 확실히 동기 실행**이라는 게 실측으로 확인됐다. 그런데
`Engine.__init__`의 오버로드 시그니처 어디에도 "async 모드로 만들어라"는 명시적
인자가 없다(`mode` 문자열 하나뿐, 유효한 값 목록은 help()에 안 나옴) - 이 프로젝트가
실제로 쓰는 `mode="gpu"`(`core/ckks_engine.py`)가 동기인지 비동기인지는 **GPU에서
직접 `Engine(mode="gpu").sync()`를 호출해봐야 확정 가능하다 - 아직 GPU를 확보 못해
미확인**. 에러 메시지가 "Async GPU mode가 아니다"라고 구체적으로 말하는 걸 보면
"Async GPU mode"라는 게 존재는 하되 `mode="gpu"`(sync)와는 다른 별도 모드일
가능성이 있다 - 즉 이 프로젝트가 쓰는 기본 gpu 모드는 이미 동기(=지금까지의
wall-clock 측정이 신뢰할 만함)일 가능성이 있지만, 이건 **추정이지 확정이 아니다**.

`PhaseTimer`는 이 불확실성에 안전하게 대응한다: `sync()`가 있으면 불러보되
"Async GPU mode가 아님" RuntimeError는 잡아서 "이미 동기 모드라 sync 불필요"로
해석하고 넘어간다(다른 종류의 예외는 그대로 전파). `sync_available`/`sync_error`
필드에 실제로 어떤 경로를 탔는지 기록해서, 보고할 때 "동기화가 실제로 적용된
측정인지 추정인지"를 구분할 수 있게 한다."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


class CountingEngineProxy:
    """ctx.engine을 감싸서 연산 호출 수를 센다. `__getattr__`로 나머지 전부를
    실제 engine에 위임하므로, 기존 코드는 `ctx.engine = CountingEngineProxy(ctx.engine)`
    한 줄만 넣으면 계측이 시작된다(baseline/packed의 실제 forward/backward 로직은
    전혀 안 건드림).

    **ct-ct vs ct-pt/스칼라 multiply 구분 방법(추정에 근거)**: 이 코드베이스의
    `ctx.engine.multiply` 호출은 전부(`tree_ops_packed.py`/`block_ops.py`/
    `gate.py`/`softmax.py`를 grep으로 확인) ct×ct에는 3번째 위치 인자로 relinearization
    key(`ctx.rlk`)를 넘기고, ct×plaintext/스칼라에는 두 인자만 쓴다는 **호출 관례**를
    따른다 - desilofhe 자체가 타입으로 구분해주는 게 아니라(`help(Engine.multiply)`의
    오버로드 목록에 ct-ct는 rlk 있는/없는 버전이 둘 다 있어, 이론적으로는 rlk 없이
    ct-ct를 곱하는 것도 API상 가능하다), 이 프로젝트가 실제로 그 오버로드를 쓴 적이
    없다는 걸 grep으로 확인한 것에 의존한다. 앞으로 누가 rlk 없이 ct-ct multiply를
    호출하는 코드를 추가하면 이 카운터가 ct-pt로 잘못 센다 - 그런 코드가 있는지는
    grep으로 한 번씩 재확인할 것."""

    def __init__(self, engine):
        self._engine = engine
        self.counts: dict[str, int] = defaultdict(int)
        self.n_refreshed_ciphertexts = 0  # bootstrap 1회당 +1, merge_bootstrap 1회당 +2

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def multiply(self, a, b, *rest, **kwargs):
        if rest or "relinearization_key" in kwargs:
            self.counts["multiply_ct_ct"] += 1
        else:
            self.counts["multiply_ct_pt_or_scalar"] += 1
        return self._engine.multiply(a, b, *rest, **kwargs)

    def rotate(self, *args, **kwargs):
        self.counts["rotate"] += 1
        return self._engine.rotate(*args, **kwargs)

    def sum(self, *args, **kwargs):
        # engine.sum()은 내부적으로 log2(slot_count)회 rotate+add를 한다 - "외부 호출
        # 횟수"만 세고 내부 rotate 횟수는 별도로 추정하지 않는다(engine 내부 구현에
        # 접근 불가 - opt/engine_counter.py와 같은 원칙).
        self.counts["sum"] += 1
        return self._engine.sum(*args, **kwargs)

    def evaluate_polynomial(self, *args, **kwargs):
        # 마찬가지로 "몇 차 다항식을 몇 번 평가했는가"의 호출 횟수만 - 내부 곱셈
        # 횟수(Paterson-Stockmeyer 등 알고리즘에 따라 다름)는 추정하지 않는다.
        self.counts["evaluate_polynomial"] += 1
        return self._engine.evaluate_polynomial(*args, **kwargs)

    def bootstrap(self, *args, **kwargs):
        self.counts["bootstrap"] += 1
        self.n_refreshed_ciphertexts += 1
        return self._engine.bootstrap(*args, **kwargs)

    def merge_bootstrap(self, *args, **kwargs):
        self.counts["merge_bootstrap"] += 1
        self.n_refreshed_ciphertexts += 2
        return self._engine.merge_bootstrap(*args, **kwargs)


@dataclass
class PhaseTimer:
    """forward/backward/parameter_update처럼 큰 단계 구간의 wall-clock 시간을 잰다.
    `ctx.engine.sync()`가 있으면(현재 desilofhe 버전에서 확인됨) 각 구간 시작/종료
    시점에 불러서 비동기 큐잉으로 인한 시간 누락을 막으려 시도한다. 다만 CPU 모드
    실측으로 확인된 대로 엔진이 애초에 동기 모드면 `sync()`가 `RuntimeError: Engine
    is not in Async GPU mode`를 던진다 - 이 경우는 "이미 동기라 sync 불필요"로 보고
    잡아서 넘어간다(다른 예외는 그대로 전파해서 진짜 오류를 숨기지 않는다).
    `sync_available`(메서드 존재 여부)과 `sync_actually_worked`(실제로 예외 없이
    성공했는지)를 따로 기록해서, 보고할 때 측정값이 진짜 sync된 것인지 추정인지
    구분한다."""

    phases: dict = field(default_factory=lambda: defaultdict(float))
    sync_available: bool | None = None
    sync_actually_worked: bool | None = None
    _t0: float | None = None
    _current: str | None = None

    def _sync(self, engine) -> None:
        has_sync = hasattr(engine, "sync")
        if self.sync_available is None:
            self.sync_available = has_sync
        if not has_sync:
            return
        try:
            engine.sync()
            self.sync_actually_worked = True
        except RuntimeError as exc:
            if "Async GPU mode" not in str(exc):
                raise
            self.sync_actually_worked = False

    def start(self, engine, phase: str) -> None:
        self._sync(engine)
        self._current = phase
        self._t0 = time.time()

    def stop(self, engine) -> None:
        if self._current is None:
            return
        self._sync(engine)
        self.phases[self._current] += time.time() - self._t0
        self._current = None


@dataclass
class EpochMeasurement:
    """한 epoch의 측정 결과 전체 - JSON으로 남겨서 여러 epoch/여러 최적화 단계 사이
    비교에 쓴다."""

    epoch: int
    phase_times_s: dict
    op_counts: dict
    n_refreshed_ciphertexts: int
    sync_available: bool | None
    sync_actually_worked: bool | None
    wall_time_s: float
    process_overhead_s: float  # 전체 wall_time_s - (setup 이후 epoch_worker 실행 자체 시간). key/data 로드+저장 포함.
    peak_gpu_mib: int | None
    param_file_bytes: int | None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "phase_times_s": dict(self.phase_times_s),
            "op_counts": dict(self.op_counts),
            "n_refreshed_ciphertexts": self.n_refreshed_ciphertexts,
            "sync_available": self.sync_available,
            "sync_actually_worked": self.sync_actually_worked,
            "wall_time_s": self.wall_time_s,
            "process_overhead_s": self.process_overhead_s,
            "peak_gpu_mib": self.peak_gpu_mib,
            "param_file_bytes": self.param_file_bytes,
            "notes": self.notes,
        }


def params_dir_total_bytes(params_dir: Path) -> int:
    return sum(p.stat().st_size for p in params_dir.glob("*.ct"))


def append_measurement_jsonl(path: Path, measurement: EpochMeasurement) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(measurement.to_dict()) + "\n")
