"""`epoch_worker_packed.py`의 grouped(wide) 버전 - `n_features*block_size > slot_count`라
feature 전체를 ciphertext 하나에 못 담는 넓은 데이터셋(예: soybean, 35 feature) 전용.
setup/finalize는 baseline 그대로 재사용(params[] 포맷이 동일 - threshold도 baseline과 같은
feature별 개별 ciphertext 리스트, feature grouping은 packed 내부 계산에서만 쓰이고 저장
포맷에는 전혀 안 드러난다). `epoch_worker_packed.py`(단일 ciphertext 버전, iris/wine/
breast_cancer가 실제로 쓰는 경로)는 이 파일 때문에 전혀 안 바뀐다."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data.serialization import load_server_context, read_dataset  # noqa: E402
from core.ckks_engine import create_bootstrap_engine  # noqa: E402
from core.runtime.gpu import GpuPeakWatcher  # noqa: E402
from models.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    assert_layout_fits,
    compute_block_size_tight,
    sample_mask_plain,
)
from models.gradient_soft_tree.packed.wide_features import (  # noqa: E402
    build_block_masks_local,
    pack_dataset_features_wide,
    plan_feature_groups,
    sample_mask_blocked_wide,
)
from models.gradient_soft_tree.packed.instrumentation import (  # noqa: E402
    CountingEngineProxy,
    EpochMeasurement,
    append_measurement_jsonl,
    params_dir_total_bytes,
)
from models.gradient_soft_tree.packed.tree_ops_packed_wide import forward_backward_update_N_packed_wide  # noqa: E402
from models.gradient_soft_tree.params import (  # noqa: E402
    load_alpha,
    load_leaf_logits,
    load_threshold,
    save_alpha,
    save_leaf_logits,
    save_threshold,
)

MEASUREMENTS_FILENAME = "measurements.jsonl"


def _next_epoch_idx(measurements_path: Path) -> int:
    if not measurements_path.exists():
        return 1
    with open(measurements_path) as f:
        return sum(1 for _ in f) + 1


def main() -> None:
    session_dir = Path(sys.argv[1])
    process_t0 = time.time()
    config = json.loads((session_dir / "config.json").read_text())
    n_features = config["n_features"]
    depth = config["depth"]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth

    engine = create_bootstrap_engine(
        mode=config["mode"], device_id=config["device_id"], level_preset=config.get("level_preset")
    )
    counter = CountingEngineProxy(engine)
    ctx = load_server_context(counter, session_dir / "keys", mode=config["mode"], device_id=config["device_id"])
    dataset = read_dataset(
        ctx,
        session_dir / "dataset",
        n_features=n_features,
        n_classes=config["n_classes"],
        n_samples=config["n_samples"],
    )
    sample_mask = sample_mask_plain(dataset.n_samples, engine.slot_count)

    block_size = compute_block_size_tight(dataset.n_samples)
    groups = plan_feature_groups(n_features, block_size, engine.slot_count)
    max_per_group = max(end - start for start, end in groups)
    assert_layout_fits(max_per_group, block_size, engine.slot_count)  # 그룹 하나가 제대로 들어가는지 재확인
    block_masks_local = build_block_masks_local(max_per_group, block_size, engine.slot_count)
    blocked_features_groups = pack_dataset_features_wide(ctx, dataset.enc_features, block_size, groups)
    sample_mask_blocked_groups = sample_mask_blocked_wide(dataset.n_samples, groups, block_size, engine.slot_count)

    params_dir = session_dir / "params"
    params = {
        "alpha": load_alpha(engine, params_dir, n_internal),
        "threshold": load_threshold(engine, params_dir, n_internal, n_features),
        "leaf_logits": load_leaf_logits(engine, params_dir, n_leaves),
    }

    epoch_t0 = time.time()
    with GpuPeakWatcher() as gpu_watcher:
        new_params = forward_backward_update_N_packed_wide(
            ctx, dataset, params, sample_mask, blocked_features_groups, sample_mask_blocked_groups,
            block_masks_local, block_size, n_features, config["n_classes"], depth, lr=config["lr"], groups=groups,
        )
    epoch_elapsed = time.time() - epoch_t0

    save_alpha(engine, params_dir, new_params["alpha"])
    save_threshold(engine, params_dir, new_params["threshold"])
    save_leaf_logits(engine, params_dir, new_params["leaf_logits"])

    measurements_path = session_dir / MEASUREMENTS_FILENAME
    epoch_idx = _next_epoch_idx(measurements_path)
    measurement = EpochMeasurement(
        epoch=epoch_idx,
        phase_times_s={"epoch_total": epoch_elapsed},
        op_counts=dict(counter.counts),
        n_refreshed_ciphertexts=counter.n_refreshed_ciphertexts,
        sync_available=None,
        sync_actually_worked=None,
        wall_time_s=time.time() - process_t0,
        process_overhead_s=(time.time() - process_t0) - epoch_elapsed,
        peak_gpu_mib=gpu_watcher.peak,
        param_file_bytes=params_dir_total_bytes(params_dir),
        notes=[
            f"n_features={n_features}가 {len(groups)}개 그룹({groups})으로 나뉨 - "
            "epoch_worker_packed.py(단일 ciphertext)와 달리 threshold packing/sigmoid가 "
            "그룹 수만큼 반복된다.",
        ],
    )
    append_measurement_jsonl(measurements_path, measurement)


if __name__ == "__main__":
    main()
