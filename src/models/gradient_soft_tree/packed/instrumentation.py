"""packed 학습(joint training, vanilla GD)의 forward/backward/parameter-update를
계측하기 위한 도구.

`CountingEngineProxy`/`PhaseTimer`는 이번 최적화 작업이 요구하는 지표(ct-ct vs ct-pt
multiply 구분, merge_bootstrap 카운트, refresh된 ciphertext 개수, phase별 GPU-sync
기반 wall time)를 위해 새로 작성했다.

`BootstrapProfiler`/`ensure_level_profiled`/`batch_ensure_level_profiled`는 2026-09-22
opt 계보 제거 때 `opt/profiler.py`에서 그대로(로직 변경 없이) 옮겨왔다 - opt의 TreeConfig
실험(현재 지원 범위 밖)과는 무관하게 이 세 가지는 범용 bootstrap 태깅/집계 도구라 packed
계측에 그대로 재사용할 수 있다.

**GPU 비동기 실행 여부 - 확정됨 (2026-09-27, GPU에서 직접 실측)**: 설치된
`desilofhe-cu130==1.14.1`의 `Engine`에 인자 없는 `sync() -> None` 메서드가 있다.
2026-09-21에는 CPU 모드(`Engine(mode="cpu")`)에서만 `RuntimeError: Engine is not
in Async GPU mode`를 확인했고 `mode="gpu"`는 미확인이었는데, 2026-09-27 실제 GPU
엔진(`create_bootstrap_context(mode="gpu", level_preset=17)`)에서 `.sync()`를 직접
호출해본 결과 **`mode="gpu"`도 똑같이 `RuntimeError: Engine is not in Async GPU
mode`를 낸다 - 이 프로젝트가 쓰는 gpu 모드는 동기 실행이 확정**됐다. 즉 지금까지
(이번 세션 전체, opt/profiler.py 시절 포함) `time.time()`으로 잰 wall-clock
측정치들은 전부 신뢰할 수 있다 - 별도 sync 없이도 각 engine 호출이 실제로 끝날
때까지 블록한다.

`PhaseTimer`는 그래도 안전하게 동작한다: `sync()`가 있으면 불러보되 "Async GPU
mode가 아님" RuntimeError는 잡아서 "이미 동기 모드라 sync 불필요"로 해석하고
넘어간다(다른 종류의 예외는 그대로 전파). `sync_available`/`sync_actually_worked`
필드에 실제로 어떤 경로를 탔는지 기록한다(이 프로젝트에서는 두 값 다 True/False로
"이미 동기라 sync 불필요"를 나타내는 게 정상 - Async 모드를 쓸 계획이 없는 한
`sync_actually_worked=False`가 오류가 아니라 기대값이다)."""

from __future__ import annotations

import csv
import json
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(measurement.to_dict()) + "\n")


# ---- 2026-09-22: opt/profiler.py에서 로직 변경 없이 이동(opt 계보 제거) ----
# tag별 bootstrap 이벤트(호출 전/후 level, 소요시간) 기록 + merge_bootstrap 페어링
# helper. Phase 5A/5B(opt 실험)가 원래 목적이었지만 이 세 가지 자체는 TreeConfig와
# 무관한 범용 bootstrap 계측이라 packed에 그대로 재사용한다.

CATEGORIES = [
    "sigmoid_forward", "attention_softmax", "leaf_softmax", "routing_forward",
    "loss", "backward_leaf", "backward_threshold", "backward_attention",
    "parameter_update", "other",
]


@dataclass
class BootstrapEvent:
    tag: str
    level_before: int
    level_after: int
    elapsed_s: float
    seq: int


@dataclass
class BootstrapProfiler:
    detailed: bool = False
    events: list = field(default_factory=list)
    all_checks: list = field(default_factory=list)
    _seq: int = 0
    _t_start: float = field(default_factory=time.time)

    def record(self, tag: str, level_before: int, level_after: int, elapsed_s: float) -> None:
        self._seq += 1
        ev = BootstrapEvent(tag=tag, level_before=level_before, level_after=level_after, elapsed_s=elapsed_s, seq=self._seq)
        self.events.append(ev)
        if self.detailed:
            print(
                f"[bootstrap #{self._seq:03d}] tag={tag:<22} level {level_before} -> {level_after}  "
                f"elapsed={elapsed_s:.3f}s",
                flush=True,
            )

    def trace(self, msg: str) -> None:
        if self.detailed:
            print(f"[trace] {msg}", flush=True)

    def summary(self) -> dict:
        by_cat = defaultdict(lambda: {"count": 0, "total_time": 0.0, "levels_before": [], "levels_after": []})
        for ev in self.events:
            d = by_cat[ev.tag]
            d["count"] += 1
            d["total_time"] += ev.elapsed_s
            d["levels_before"].append(ev.level_before)
            d["levels_after"].append(ev.level_after)
        out = {}
        for tag, d in by_cat.items():
            n = d["count"]
            out[tag] = {
                "count": n,
                "total_time_s": d["total_time"],
                "avg_time_s": d["total_time"] / n if n else 0.0,
                "avg_level_before": sum(d["levels_before"]) / n if n else 0.0,
                "avg_level_after": sum(d["levels_after"]) / n if n else 0.0,
            }
        return out

    def print_summary(self) -> None:
        s = self.summary()
        total_count = sum(v["count"] for v in s.values())
        total_time = sum(v["total_time_s"] for v in s.values())
        wall = time.time() - self._t_start
        print("\nBootstrap Profile")
        print(f"{'category':<22}{'count':>7}{'total_s':>10}{'avg_s':>8}{'lvl_before':>12}{'lvl_after':>11}")
        for tag in CATEGORIES:
            if tag not in s:
                continue
            v = s[tag]
            print(
                f"{tag:<22}{v['count']:>7}{v['total_time_s']:>10.1f}{v['avg_time_s']:>8.2f}"
                f"{v['avg_level_before']:>12.1f}{v['avg_level_after']:>11.1f}"
            )
        for tag, v in s.items():
            if tag not in CATEGORIES:
                print(
                    f"{tag:<22}{v['count']:>7}{v['total_time_s']:>10.1f}{v['avg_time_s']:>8.2f}"
                    f"{v['avg_level_before']:>12.1f}{v['avg_level_after']:>11.1f}"
                )
        print(f"{'TOTAL':<22}{total_count:>7}{total_time:>10.1f}")
        print(f"(wall time so far: {wall:.1f}s, bootstrap share: {100.0 * total_time / wall:.1f}%)")

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps({"summary": self.summary(), "n_events": len(self.events)}, indent=2))

    def events_to_csv(self, path: Path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seq", "tag", "level_before", "level_after", "elapsed_s"])
            for ev in self.events:
                w.writerow([ev.seq, ev.tag, ev.level_before, ev.level_after, f"{ev.elapsed_s:.4f}"])

    def checks_summary(self) -> dict:
        by_tag = defaultdict(lambda: {"n": 0, "n_triggered": 0, "levels": []})
        for tag, level, triggered in self.all_checks:
            d = by_tag[tag]
            d["n"] += 1
            d["n_triggered"] += int(triggered)
            d["levels"].append(level)
        out = {}
        for tag, d in by_tag.items():
            levels = d["levels"]
            out[tag] = {
                "n_checks": d["n"],
                "n_triggered": d["n_triggered"],
                "trigger_rate": d["n_triggered"] / d["n"] if d["n"] else 0.0,
                "level_min": min(levels) if levels else None,
                "level_mean": sum(levels) / len(levels) if levels else None,
                "level_max": max(levels) if levels else None,
            }
        return out

    def checks_to_csv(self, path: Path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tag", "level_before", "triggered"])
            for tag, level, triggered in self.all_checks:
                w.writerow([tag, level, int(triggered)])


def batch_ensure_level_profiled(ctx, items: list, min_level: int, profiler: "BootstrapProfiler | None"):
    """merge_bootstrap 페어링 패턴(2개씩 묶어 한 번에 복구, 홀수면 마지막 하나만 단독
    bootstrap) + tag/시간 profiling. items: [(ciphertext, tag), ...]. 반환은 같은
    순서의 새 ciphertext 리스트."""
    result = [ctx.engine.intt(ct) for ct, _tag in items]
    tags = [tag for _ct, tag in items]
    needs_refresh = [i for i, ct in enumerate(result) if ct.level < min_level]
    i = 0
    while i < len(needs_refresh):
        if i + 1 < len(needs_refresh):
            idx1, idx2 = needs_refresh[i], needs_refresh[i + 1]
            level_before = min(result[idx1].level, result[idx2].level)
            t0 = time.time()
            r1, r2 = ctx.engine.merge_bootstrap(
                result[idx1], result[idx2], ctx.rlk, ctx.conjugation_key, ctx.rotation_key, ctx.small_bootstrap_key
            )
            elapsed = time.time() - t0
            result[idx1], result[idx2] = ctx.engine.intt(r1), ctx.engine.intt(r2)
            if profiler is not None:
                profiler.record(f"{tags[idx1]}|{tags[idx2]}_merged", level_before, result[idx1].level, elapsed)
            i += 2
        else:
            idx = needs_refresh[i]
            level_before = result[idx].level
            t0 = time.time()
            refreshed = ctx.engine.bootstrap(
                result[idx], ctx.rlk, ctx.conjugation_key, ctx.rotation_key, ctx.small_bootstrap_key
            )
            elapsed = time.time() - t0
            result[idx] = ctx.engine.intt(refreshed)
            if profiler is not None:
                profiler.record(f"{tags[idx]}_solo", level_before, result[idx].level, elapsed)
            i += 1
    return result


def ensure_level_profiled(ctx, ct, tag: str, min_level: int, profiler: "BootstrapProfiler | None"):
    """core.ckks_engine.ensure_level()과 완전히 동일한 로직(intt 정규화 + min_level
    guard) + tag가 붙은 profiler 기록."""
    level_before = ct.level
    triggered = ct.level < min_level
    if profiler is not None:
        profiler.trace(f"{tag} input level: {level_before}")
        profiler.all_checks.append((tag, level_before, triggered))
    if triggered:
        ct = ctx.engine.intt(ct)
        t0 = time.time()
        refreshed = ctx.engine.bootstrap(
            ct, ctx.rlk, ctx.conjugation_key, ctx.rotation_key, ctx.small_bootstrap_key
        )
        elapsed = time.time() - t0
        result = ctx.engine.intt(refreshed)
        if profiler is not None:
            profiler.record(tag, level_before, result.level, elapsed)
            profiler.trace(f"{tag} output level: {result.level}")
        return result
    return ct
