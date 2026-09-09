"""epoch_worker_N.py와 같은 프로세스 격리 패턴이지만 forward_backward_update_N_opt(설정
가능한 버전)을 쓴다. session_dir/experiment_config.json에서 TreeConfig를 읽는다.

매 epoch마다 새 BootstrapProfiler로 그 epoch의 bootstrap 이벤트를 기록해서
session_dir/profile/epoch_{n}.csv에 저장한다 (프로세스가 끝나면 죽으므로 in-memory 누적이
불가능 - 파일로 넘긴다)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from closed_form_mgi.io_utils import load_context, read_dataset  # noqa: E402
from closed_form_mgi.primitives import create_bootstrap_engine  # noqa: E402
from experiments.gradient_soft_tree.opt.config import TreeConfig  # noqa: E402
from experiments.gradient_soft_tree.opt.engine_counter import CountingEngineProxy  # noqa: E402
from experiments.gradient_soft_tree.opt.profiler import BootstrapProfiler  # noqa: E402
from experiments.gradient_soft_tree.opt.tree_ops import forward_backward_update_N_opt  # noqa: E402

import re


def _next_epoch_idx(profile_dir: Path) -> int:
    """profile_dir에 이미 있는 'epoch_<N>.csv'(정확히 이 패턴만 - 'epoch_*.csv' glob은
    'epoch_11_checks.csv'도 매칭해버려서 매 epoch 파일이 2개씩 늘어나 번호가 두 배로
    밀리는 버그가 있었다, 2026-08-27 실측으로 발견) 파일 수 + 1을 다음 epoch 번호로 쓴다."""
    if not profile_dir.exists():
        return 1
    pattern = re.compile(r"^epoch_(\d+)\.csv$")
    matches = [pattern.match(p.name) for p in profile_dir.iterdir()]
    return sum(1 for m in matches if m) + 1


def main() -> None:
    session_dir = Path(sys.argv[1])
    detailed = len(sys.argv) > 2 and sys.argv[2] == "detailed"
    config = json.loads((session_dir / "config.json").read_text())
    exp = json.loads((session_dir / "experiment_config.json").read_text())
    tree_config = TreeConfig(**exp)

    n_features = config["n_features"]
    depth = config["depth"]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth

    engine = create_bootstrap_engine(
        mode=config["mode"], device_id=config["device_id"], level_preset=config.get("level_preset")
    )
    ctx = load_context(engine, session_dir / "keys", mode=config["mode"], device_id=config["device_id"])
    dataset = read_dataset(
        ctx, session_dir / "dataset", n_features=n_features, n_classes=config["n_classes"], n_samples=config["n_samples"]
    )
    sample_mask = engine.read_ciphertext(session_dir / "sample_mask.ct")

    params_dir = session_dir / "params"
    params = {
        "alpha": [engine.read_ciphertext(params_dir / f"alpha_{i}.ct") for i in range(n_internal)],
        "threshold": [
            [engine.read_ciphertext(params_dir / f"threshold_{i}_{j}.ct") for j in range(n_features)]
            for i in range(n_internal)
        ],
        "leaf_logits": [engine.read_ciphertext(params_dir / f"leaf_{l}.ct") for l in range(n_leaves)],
    }

    counter = CountingEngineProxy(engine)
    ctx.engine = counter  # bootstrap() 자체는 profiler.ensure_level_profiled가 ctx.engine.bootstrap을
    # 직접 호출하므로 이 카운터로도 같이 잡힌다 - profiler와 이중 계측이지만 서로 다른 관점
    # (profiler=tag/level/시간, counter=단순 총 호출수)이라 상호 검증에도 쓸 수 있다.

    def _param_level_snapshot() -> dict:
        """Phase 5B: decrypt 없이 .level만 읽어서 alpha/threshold/leaf_logits 분포를 기록."""
        def stats(levels):
            return {"min": min(levels), "mean": sum(levels) / len(levels), "max": max(levels)}
        alpha_levels = [ct.level for ct in params["alpha"]]
        threshold_levels = [ct.level for row in params["threshold"] for ct in row]
        leaf_levels = [ct.level for ct in params["leaf_logits"]]
        return {"alpha": stats(alpha_levels), "threshold": stats(threshold_levels), "leaf_logits": stats(leaf_levels)}

    level_trace_dir = session_dir / "level_trace"
    level_trace_dir.mkdir(parents=True, exist_ok=True)
    epoch_idx_for_trace = _next_epoch_idx(session_dir / "profile")
    level_at_start = _param_level_snapshot()

    profiler = BootstrapProfiler(detailed=detailed)
    t0 = time.time()
    new_params = forward_backward_update_N_opt(
        ctx, dataset, params, sample_mask, n_features, config["n_classes"], depth, lr=config["lr"],
        config=tree_config, profiler=profiler,
    )
    elapsed = time.time() - t0
    op_counts = dict(counter.counts)

    def _new_param_level_snapshot() -> dict:
        def stats(levels):
            return {"min": min(levels), "mean": sum(levels) / len(levels), "max": max(levels)}
        alpha_levels = [ct.level for ct in new_params["alpha"]]
        threshold_levels = [ct.level for row in new_params["threshold"] for ct in row]
        leaf_levels = [ct.level for ct in new_params["leaf_logits"]]
        return {"alpha": stats(alpha_levels), "threshold": stats(threshold_levels), "leaf_logits": stats(leaf_levels)}

    level_at_end = _new_param_level_snapshot()
    (level_trace_dir / f"epoch_{epoch_idx_for_trace}.json").write_text(
        json.dumps({"start": level_at_start, "end": level_at_end}, indent=2)
    )

    for i in range(n_internal):
        engine.write_ciphertext(new_params["alpha"][i], params_dir / f"alpha_{i}.ct")
        for j in range(n_features):
            engine.write_ciphertext(new_params["threshold"][i][j], params_dir / f"threshold_{i}_{j}.ct")
    for l in range(n_leaves):
        engine.write_ciphertext(new_params["leaf_logits"][l], params_dir / f"leaf_{l}.ct")

    profile_dir = session_dir / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    epoch_idx = _next_epoch_idx(profile_dir)
    profiler.events_to_csv(profile_dir / f"epoch_{epoch_idx}.csv")
    profiler.checks_to_csv(profile_dir / f"epoch_{epoch_idx}_checks.csv")

    profiler.print_summary()
    s = profiler.summary()
    checks_summary = profiler.checks_summary()
    total_bootstrap_time = sum(v["total_time_s"] for v in s.values())
    total_bootstrap_count = sum(v["count"] for v in s.values())
    result = {
        "epoch_elapsed_s": elapsed,
        "bootstrap_count": total_bootstrap_count,
        "bootstrap_time_s": total_bootstrap_time,
        "bootstrap_share": total_bootstrap_time / elapsed if elapsed > 0 else 0.0,
        "by_category": s,
        "op_counts": op_counts,
        "checks_summary": checks_summary,
        "level_trace": {"start": level_at_start, "end": level_at_end},
    }
    (profile_dir / f"epoch_{epoch_idx}_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"epoch_elapsed_s": elapsed, "bootstrap_count": total_bootstrap_count, "bootstrap_time_s": total_bootstrap_time}))


if __name__ == "__main__":
    main()
