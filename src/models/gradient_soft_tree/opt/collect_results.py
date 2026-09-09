"""results/manifest.jsonl에 기록된 모든 run을 모아 두 개의 요구 CSV를 만든다:
  results/bootstrap_optimization_summary.csv
  results/bootstrap_by_module.csv

python -m models.gradient_soft_tree.opt.collect_results
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "results" / "manifest.jsonl"

# Phase 0 카테고리 -> 요구 표의 module 열 매핑 (profiler.CATEGORIES 참고)
MODULE_MAP = {
    "sigmoid_forward": "Sigmoid",
    "attention_softmax": "Attention",
    "leaf_softmax": "Leaf",
    "routing_forward": "Backward",  # forward routing은 아래에서 별도 열로 안 나눔(요구 표에 없음) - Other로 재배치
    "loss": "Backward",
    "backward_leaf": "Backward",
    "backward_gate": "Backward",
    # 2026-08-27: Phase 5 진단을 위해 backward_threshold/backward_attention을 call-site
    # 단위로 더 세분화(tree_ops.py) - 접두어로 매칭해서 기존 module 버킷 그대로 유지.
    "backward_threshold": "Backward",
    "backward_threshold_gatej": "Backward",
    "backward_threshold_surrogate": "Backward",
    "backward_threshold_sumt": "Backward",
    "backward_threshold_dLdtj": "Backward",
    "backward_attention": "Backward",
    "backward_attention_wj": "Backward",
    "backward_attention_suma": "Backward",
    "backward_attention_combine": "Backward",
    "parameter_update": "Other",
    "adam_first_moment": "Adam",
    "adam_second_moment": "Adam",
    "adam_rsqrt_or_reciprocal": "Adam",
    "other": "Other",
}
MODULE_COLS = ["Sigmoid", "Attention", "Leaf", "Backward", "Adam", "Other"]


def load_manifest_rows() -> list[dict]:
    if not MANIFEST.exists():
        return []
    rows = []
    for line in MANIFEST.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_result(session_dir: str) -> dict | None:
    p = Path(session_dir) / "result.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def aggregate_profile(session_dir: str) -> tuple[dict, dict]:
    """session_dir/profile/epoch_*_summary.json 전부를 합산 -> (category별 count/time, op_counts 합)."""
    profile_dir = Path(session_dir) / "profile"
    by_cat = defaultdict(lambda: {"count": 0, "total_time_s": 0.0})
    op_counts = defaultdict(int)
    for f in sorted(profile_dir.glob("epoch_*_summary.json")):
        d = json.loads(f.read_text())
        for tag, v in d.get("by_category", {}).items():
            by_cat[tag]["count"] += v["count"]
            by_cat[tag]["total_time_s"] += v["total_time_s"]
        for k, v in d.get("op_counts", {}).items():
            op_counts[k] += v
    return dict(by_cat), dict(op_counts)


def main() -> None:
    rows = load_manifest_rows()
    if not rows:
        print("no runs recorded in results/manifest.jsonl yet")
        return

    summary_path = REPO_ROOT / "results" / "bootstrap_optimization_summary.csv"
    module_path = REPO_ROOT / "results" / "bootstrap_by_module.csv"

    with open(summary_path, "w", newline="") as f_sum, open(module_path, "w", newline="") as f_mod:
        w_sum = csv.writer(f_sum)
        w_sum.writerow([
            "Version", "Dataset", "Depth", "Epochs", "Acc(test)", "Acc(train)", "CKKS_diff",
            "Bootstrap", "Bootstrap_sec", "Epoch_sec(last)", "Mult", "Rotation", "GPU_MB(peak)", "session_dir",
        ])
        w_mod = csv.writer(f_mod)
        w_mod.writerow(["Version"] + MODULE_COLS + ["Total"])

        for row in rows:
            session_dir = row["session_dir"]
            result = load_result(session_dir)
            if result is None:
                continue
            by_cat, op_counts = aggregate_profile(session_dir)
            total_bootstrap = sum(v["count"] for v in by_cat.values())
            total_bootstrap_time = sum(v["total_time_s"] for v in by_cat.values())
            last_epoch = result["epoch_results"][-1] if result.get("epoch_results") else {}

            w_sum.writerow([
                row["preset"], row["dataset"], row["depth"], row["n_epochs"],
                f"{result.get('test_acc', float('nan')):.4f}", f"{result.get('train_acc', float('nan')):.4f}",
                f"{result.get('max_err', float('nan')):.5f}",
                total_bootstrap, f"{total_bootstrap_time:.1f}",
                f"{last_epoch.get('epoch_elapsed_s', float('nan')):.1f}",
                op_counts.get("multiply", ""), op_counts.get("rotate", "") ,
                result.get("peak_gpu_mib", ""), session_dir,
            ])

            mod_counts = defaultdict(int)
            for tag, v in by_cat.items():
                mod = MODULE_MAP.get(tag, "Other")
                mod_counts[mod] += v["count"]
            w_mod.writerow([row["preset"]] + [mod_counts.get(c, 0) for c in MODULE_COLS] + [total_bootstrap])

    print(f"wrote {summary_path}")
    print(f"wrote {module_path}")


if __name__ == "__main__":
    main()
