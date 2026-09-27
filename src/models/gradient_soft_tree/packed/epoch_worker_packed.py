"""epoch_worker_N.py를 forward_backward_update_N_packed용으로 바꾼 버전. setup은
baseline과 완전히 동일하므로 `setup_worker_N.py`를 그대로 재사용한다(packed 전용 setup
파일이 따로 필요 없음 - 데이터셋/초기 파라미터 암호화 로직은 packing과 무관). finalize도
`params[]` 포맷이 baseline과 동일해서 `finalize_worker_N.py`를 그대로 재사용한다.

이 파일만 packed 전용인 이유: `forward_backward_update_N_packed`를 부르려면
blocked_features/sample_mask_blocked/block_masks/block_size가 필요한데, 전부 (1) dataset/
sample_mask(이미 session_dir에 저장돼 setup_worker_N.py가 만들어둠)의 순수 함수이고 (2)
계산 비용이 싸므로(rotation만, bootstrap 없음) 매 epoch_worker_packed 프로세스 시작 시
새로 계산한다 - 별도로 직렬화/로드할 필요가 없다."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data.serialization import load_context, read_dataset  # noqa: E402
from core.ckks_engine import create_bootstrap_engine  # noqa: E402
from core.runtime.gpu import GpuPeakWatcher  # noqa: E402
from models.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    assert_layout_fits,
    broadcast_full_to_blocks,
    build_block_masks,
    compute_block_size,
    pack_dataset_features_blocked,
)
from models.gradient_soft_tree.packed.instrumentation import (  # noqa: E402
    CountingEngineProxy,
    EpochMeasurement,
    append_measurement_jsonl,
    params_dir_total_bytes,
)
from models.gradient_soft_tree.packed.tree_ops_packed import forward_backward_update_N_packed  # noqa: E402
from models.gradient_soft_tree.params import (  # noqa: E402
    load_alpha,
    load_leaf_logits,
    load_threshold,
    save_alpha,
    save_leaf_logits,
    save_threshold,
)

# 2026-09-27: Step 1(최적화 작업 기준값 확보) - ctx.engine을 CountingEngineProxy로 감싸서
# 호출 횟수를 세고, GpuPeakWatcher로 peak memory를 잰다. 둘 다 __getattr__로 원래 동작에
# **위임만** 하므로(값을 바꾸거나 순서를 바꾸지 않음) forward_backward_update_N_packed의
# 실제 연산에는 전혀 영향이 없다 - 계측을 껐다 켰다 해도 학습 결과(파라미터)는 완전히
# 동일해야 한다. 측정 결과는 session_dir/measurements.jsonl에 한 줄씩 누적된다.
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
    ctx = load_context(counter, session_dir / "keys", mode=config["mode"], device_id=config["device_id"])
    dataset = read_dataset(
        ctx,
        session_dir / "dataset",
        n_features=n_features,
        n_classes=config["n_classes"],
        n_samples=config["n_samples"],
    )
    sample_mask = engine.read_ciphertext(session_dir / "sample_mask.ct")

    block_size = compute_block_size(dataset.n_samples)
    assert_layout_fits(n_features, block_size, engine.slot_count)
    block_masks = build_block_masks(n_features, block_size, engine.slot_count)
    blocked_features = pack_dataset_features_blocked(ctx, dataset.enc_features, block_size)
    sample_mask_blocked = broadcast_full_to_blocks(ctx, sample_mask, n_features, block_size)

    params_dir = session_dir / "params"
    params = {
        "alpha": load_alpha(engine, params_dir, n_internal),
        "threshold": load_threshold(engine, params_dir, n_internal, n_features),
        "leaf_logits": load_leaf_logits(engine, params_dir, n_leaves),
    }

    epoch_t0 = time.time()
    with GpuPeakWatcher() as gpu_watcher:
        new_params = forward_backward_update_N_packed(
            ctx, dataset, params, sample_mask, blocked_features, sample_mask_blocked,
            block_masks, block_size, n_features, config["n_classes"], depth, lr=config["lr"],
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
            "phase_times_s는 forward/backward/parameter_update로 안 나뉨 - "
            "tree_ops_packed.py 내부는 안 건드렸고 전체 forward_backward_update_N_packed "
            "호출 하나만 쟀다(backward와 parameter_update가 코드상 한 pass에 섞여 있어 "
            "나누려면 알고리즘 구조 자체를 바꿔야 함 - 이번 계측 단계에서는 안 함).",
            "op_counts의 sum/evaluate_polynomial은 '외부 호출 횟수'다 - 그 안에서 실제로 "
            "몇 번의 rotate/multiply가 도는지(예: sum은 log2(slot_count)회 rotate)는 "
            "engine 내부 구현에 접근할 수 없어 별도로 세지 않았다(opt/engine_counter.py 때부터의 "
            "원칙과 동일).",
        ],
    )
    append_measurement_jsonl(measurements_path, measurement)


if __name__ == "__main__":
    main()
