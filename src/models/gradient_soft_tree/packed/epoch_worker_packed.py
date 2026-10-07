"""epoch_worker_N.py를 forward_backward_update_N_packed용으로 바꾼 버전. setup은
baseline과 완전히 동일하므로 `setup_worker_N.py`를 그대로 재사용한다(packed 전용 setup
파일이 따로 필요 없음 - 데이터셋/초기 파라미터 암호화 로직은 packing과 무관). finalize도
`params[]` 포맷이 baseline과 동일해서 `finalize_worker_N.py`를 그대로 재사용한다.

이 파일만 packed 전용인 이유: `forward_backward_update_N_packed`를 부르려면
blocked_features/sample_mask_blocked/block_masks/block_size가 필요한데, 전부 (1) dataset/
n_samples(config.json에 이미 공개된 값)의 순수 함수이고 (2) 계산 비용이 싸므로(rotation만,
bootstrap 없음) 매 epoch_worker_packed 프로세스 시작 시 새로 계산한다 - 별도로 직렬화/로드할
필요가 없다.

**2026-09-28 Step 2**: setup_worker_N.py가 `session_dir/sample_mask.ct`를 여전히 만들지만
(baseline 전용, 변경 없음) packed는 이제 이 파일을 읽지 않는다 - n_samples가 이미
config.json에 공개돼 있어 `block_ops.sample_mask_plain`으로 plaintext를 즉석에서 만든다."""

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
    build_block_masks,
    compute_block_size_tight,
    pack_dataset_features_blocked,
    sample_mask_blocked_plain,
    sample_mask_plain,
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
    ctx = load_server_context(counter, session_dir / "keys", mode=config["mode"], device_id=config["device_id"])
    dataset = read_dataset(
        ctx,
        session_dir / "dataset",
        n_features=n_features,
        n_classes=config["n_classes"],
        n_samples=config["n_samples"],
    )
    # 2026-09-28: Step 2 - n_samples는 config.json에 이미 공개돼 있어 sample_mask를
    # ciphertext로 저장/로드할 필요가 없다(baseline/setup_worker.py가 만든
    # sample_mask.ct는 baseline 전용 - packed는 이제 사용하지 않는다). 값/적용 위치는
    # 기존 ciphertext 버전과 동일, plaintext(numpy 배열)로 직접 구성.
    sample_mask = sample_mask_plain(dataset.n_samples, engine.slot_count)

    # 2026-09-28 Step 3: compute_block_size(2x 마진, core.encrypted_ops.block_ops 공유
    # 기본값)가 아니라 packed 전용 opt-in인 compute_block_size_tight(1x, B=next_pow2
    # (n_samples))을 쓴다 - block_size는 어떤 param 파일에도 저장되지 않는 순수 매 epoch
    # 재계산값이라(alpha/threshold/leaf_logits 포맷과 무관) 세션 호환성 문제가 없다.
    # 안전성 근거: numpy 시뮬레이션 + CPU-mode 실제 CKKS(tests/test_block_ops.py의
    # main_tight_block_size) 양쪽에서 확인 - block_local_sum의 시작 슬롯 합은 fold 전
    # sample_mask_blocked 마스킹만 되어 있으면 마진 크기와 무관하게 정확하다(마진은
    # "시작 슬롯이 아닌 다른 슬롯"의 오염만 막아주는데, 그 슬롯들은 애초에
    # gather_block_tops_to_packed가 절대 안 읽는다).
    block_size = compute_block_size_tight(dataset.n_samples)
    assert_layout_fits(n_features, block_size, engine.slot_count)
    block_masks = build_block_masks(n_features, block_size, engine.slot_count)
    blocked_features = pack_dataset_features_blocked(ctx, dataset.enc_features, block_size)
    sample_mask_blocked = sample_mask_blocked_plain(dataset.n_samples, n_features, block_size, engine.slot_count)

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
