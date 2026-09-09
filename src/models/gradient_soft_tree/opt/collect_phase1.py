"""results/softmax_iteration_ablation.csv (Phase 1 요구 포맷)을 manifest.jsonl에서 뽑아낸다.

python -m models.gradient_soft_tree.opt.collect_phase1
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from models.gradient_soft_tree.opt.collect_results import aggregate_profile, load_manifest_rows, load_result
from models.gradient_soft_tree.opt.experiments_registry import get_preset

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIP_PRESETS = {"recip14", "recip10", "recip8", "recip6", "recip4", "baseline"}


def main() -> None:
    rows = [r for r in load_manifest_rows() if r["preset"] in RECIP_PRESETS]
    out_path = REPO_ROOT / "results" / "softmax_iteration_ablation.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reciprocal_iter", "preset", "acc", "max_abs_diff", "bootstrap", "epoch_sec", "max_gpu_mib", "nan_or_inf"])
        for row in rows:
            result = load_result(row["session_dir"])
            if result is None:
                continue
            by_cat, _ = aggregate_profile(row["session_dir"])
            total_bootstrap = sum(v["count"] for v in by_cat.values())
            cfg = get_preset(row["preset"])
            last_epoch = result["epoch_results"][-1] if result.get("epoch_results") else {}
            has_nan = any(
                math.isnan(result.get(k, 0.0)) or math.isinf(result.get(k, 0.0))
                for k in ("train_acc", "test_acc", "max_err")
            )
            w.writerow([
                cfg.reciprocal_iterations, row["preset"], f"{result.get('test_acc', float('nan')):.4f}",
                f"{result.get('max_err', float('nan')):.5f}", total_bootstrap,
                f"{last_epoch.get('epoch_elapsed_s', float('nan')):.1f}",
                result.get("peak_gpu_mib", ""), has_nan,
            ])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
