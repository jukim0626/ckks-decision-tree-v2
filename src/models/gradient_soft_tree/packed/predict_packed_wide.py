"""`predict_packed.py`의 grouped(wide) 버전 - epoch_worker_packed_wide.py로 학습한
session(`n_features*block_size > slot_count`인 넓은 데이터셋)을 위한 진짜 encrypted
inference. 원리/유일한 decrypt 지점(최종 class score)은 단일 ciphertext 버전과 완전히
동일하다.

python -m models.gradient_soft_tree.packed.predict_packed_wide <session_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import (  # noqa: E402
    encrypt_dataset,
    load_scaler,
    one_hot_encode,
    scale_features,
    resolve_leaf_family_test_size,
    split_dataset_subset,
)
from core.data.serialization import load_client_secret_key, load_server_context  # noqa: E402
from core.ckks_engine import create_bootstrap_engine  # noqa: E402
from models.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    assert_layout_fits,
    compute_block_size_tight,
    sample_mask_plain,
)
from models.gradient_soft_tree.packed.wide_features import (  # noqa: E402
    build_block_masks_local,
    pack_dataset_features_wide,
    plan_feature_groups,
)
from models.gradient_soft_tree.packed.tree_ops_packed_wide import predict_packed_wide  # noqa: E402


def main() -> None:
    session_dir = Path(sys.argv[1])
    config = json.loads((session_dir / "config.json").read_text())
    n_features = config["n_features"]
    n_classes = config["n_classes"]
    depth = config["depth"]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth

    print(f"[predict_packed_wide] session={session_dir} dataset={config['dataset_name']} depth={depth} - 모델을 decrypt하지 않고 추론합니다", flush=True)

    engine = create_bootstrap_engine(
        mode=config["mode"], device_id=config["device_id"], level_preset=config.get("level_preset")
    )
    ctx = load_server_context(engine, session_dir / "keys", mode=config["mode"], device_id=config["device_id"])

    params_dir = session_dir / "params"
    params = {
        "alpha": [engine.read_ciphertext(params_dir / f"alpha_{i}.ct") for i in range(n_internal)],
        "threshold": [
            [engine.read_ciphertext(params_dir / f"threshold_{i}_{j}.ct") for j in range(n_features)]
            for i in range(n_internal)
        ],
        "leaf_logits": [engine.read_ciphertext(params_dir / f"leaf_{l}.ct") for l in range(n_leaves)],
    }

    test_size = resolve_leaf_family_test_size(config)
    X_train_raw, X_test_raw, y_train, y_test, _ = split_dataset_subset(
        config["dataset_name"], test_size=test_size, max_train=config.get("max_train")
    )
    scaler = load_scaler(
        session_dir / "client" / "scaler.json",
        expected_dataset_name=config["dataset_name"],
        expected_n_features=n_features,
    )
    X_test = scale_features(scaler, X_test_raw)
    y_test_oh = one_hot_encode(y_test, n_classes)
    dataset_test = encrypt_dataset(ctx, X_test, y_test_oh)
    sample_mask_test = sample_mask_plain(dataset_test.n_samples, engine.slot_count)

    block_size = compute_block_size_tight(dataset_test.n_samples)
    groups = plan_feature_groups(n_features, block_size, engine.slot_count)
    max_per_group = max(end - start for start, end in groups)
    assert_layout_fits(max_per_group, block_size, engine.slot_count)
    block_masks_local = build_block_masks_local(max_per_group, block_size, engine.slot_count)
    blocked_features_test_groups = pack_dataset_features_wide(ctx, dataset_test.enc_features, block_size, groups)

    print(f"[predict_packed_wide] test set encrypt 완료 (n_test={dataset_test.n_samples}, block_size={block_size}, groups={groups}) - forward 실행 중...", flush=True)

    y_hat = predict_packed_wide(
        ctx, dataset_test, params, sample_mask_test, blocked_features_test_groups,
        block_masks_local, block_size, n_features, n_classes, depth, groups=groups,
    )

    n_test = dataset_test.n_samples
    # --- 여기부터 client 측: 서버가 돌려준 y_hat을 client가 자기 sk로 decrypt ---
    client_sk = load_client_secret_key(engine, session_dir)
    scores = np.array([np.real(engine.decrypt(y_hat[c], client_sk))[:n_test] for c in range(n_classes)])
    pred = scores.argmax(axis=0)
    acc = (pred == y_test).mean()

    print(f"[predict_packed_wide] encrypted inference 완료: test_acc={acc:.4f} ({int((pred==y_test).sum())}/{n_test})", flush=True)
    print(f"[predict_packed_wide] 예측: {pred.tolist()}", flush=True)
    print(f"[predict_packed_wide] 정답: {y_test.tolist()}", flush=True)


if __name__ == "__main__":
    main()
