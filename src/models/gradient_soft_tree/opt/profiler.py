"""Phase 0: bootstrap call-site profiling.

closed_form_mgi/primitives.ensure_level()는 ct.level이 min_level 밑으로 떨어졌을 때만
실제로 bootstrap()을 호출한다. 이 모듈은 그 판단 로직을 그대로 재현하면서, 실제로
bootstrap이 발생한 호출마다 (reason tag, 호출 전 level, 호출 후 level, 소요시간)을
기록한다. baseline의 ensure_level 자체는 건드리지 않고, opt/tree_ops.py가 이 모듈의
`ensure_level_profiled()`를 baseline의 `_ensure_level` 대신 호출하는 방식으로 계측한다.

summary mode: category별 집계만 (기본, 출력 폭발 방지)
detailed mode: 매 bootstrap 호출의 trace를 순서대로 기록 (node0 sigmoid input level: ... 형태)
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Phase 0 문서가 요구한 카테고리에서, 실제 production 코드(depthN_ckks.py)가 vanilla GD를
# 쓰고 Adam은 CKKS로 아직 포팅되지 않았다는 사실(CLAUDE.md/EXPERIMENT_LOG.md 2026-08-27,
# [[project-ckks-decision-tree]])에 맞춰 카테고리를 조정했다. adam_* 카테고리는 Phase 7에서
# opt/momentum 실험을 만들 때만 실제로 값이 생긴다 - 지금 baseline profiling에서는 0이 나오는
# 게 정상이며 버그가 아니다.
CATEGORIES = [
    "sigmoid_forward",       # gate poly(sigmoid 또는 native low-degree gate) 진입 전 guard
    "attention_softmax",     # 노드 feature-attention softmax(packed_softmax) forward
    "leaf_softmax",          # leaf class-distribution softmax(packed_softmax) forward
    "routing_forward",       # gate 합산/left-right reach probability 전파
    "loss",                  # dL/dyhat 계산
    "backward_leaf",         # leaf backward (softmax_backward_packed 포함)
    "backward_threshold",    # threshold gradient (per-feature)
    "backward_attention",    # alpha gradient (per-feature + 최종 결합)
    "parameter_update",      # SGD/momentum 파라미터 갱신 직후 guard
    "adam_first_moment",
    "adam_second_moment",
    "adam_rsqrt_or_reciprocal",
    "other",
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
    # Phase 5B: 실제로 bootstrap이 발동했는지와 무관하게 모든 ensure_level "체크" 자체를
    # (tag, level, triggered) 3-tuple로 가볍게 기록한다. events는 발동한 것만(=113개
    # 규모)이라 이미 저장 비용이 작았지만, all_checks는 호출 "사이트" 전부(수백~천 단위)라
    # tuple 리스트로만 유지해서 오버헤드를 최소화한다 - Phase 5A(어디를 hoist할지)와
    # Phase 5B(레벨 분포와 bootstrap spike 상관관계)를 실측으로 판단하기 위한 원자료.
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
        """비-bootstrap level trace (예: 'node0 sigmoid input level: 15'). detailed mode에서만 출력."""
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
        """tag별 level_before 분포(min/mean/max) + trigger율. all_checks가 비어있으면(구버전
        호출부) 빈 dict."""
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
    """Phase 5A: closed_form_mgi/simd_argmin.batch_ensure_level와 같은 merge_bootstrap
    페어링 패턴(2개씩 묶어 한 번에 복구, 홀수면 마지막 하나만 단독 bootstrap) + tag/시간
    profiling. items: [(ciphertext, tag), ...]. 반환은 같은 순서의 새 ciphertext 리스트.

    engine.bootstrap()과 달리 merge_bootstrap()은 그 자체로 이미 "2개를 한 번에" 처리하는
    연산이라, tag를 두 항목의 tag를 합쳐 하나의 이벤트로 기록한다(개별 bootstrap과 직접
    비교 가능하도록 profiler.events에는 여전히 1개 이벤트로 남는다 - "몇 번 GPU 호출을
    했는가"가 실제 wall-clock 비용에 대응하는 지표이기 때문)."""
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
    """closed_form_mgi.primitives.ensure_level()과 완전히 동일한 로직(intt 정규화 +
    min_level guard) + tag가 붙은 profiler 기록. baseline ensure_level은 안 건드림."""
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
