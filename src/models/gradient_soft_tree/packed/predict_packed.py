"""학습된(암호화 상태 그대로 저장된) session_dir을 읽어서, **모델을 한 번도 decrypt하지
않고** test set에 대해 진짜 encrypted inference를 수행한다. `finalize_worker_N.py`가
지금까지 계산해온 train_acc/test_acc는 사실 "params를 decrypt한 뒤 평문으로 predict"한
것(검증용)이었고, 이 스크립트가 이 프로젝트의 목표("학습과 추론 모두를 암호화 상태로
수행", CLAUDE.md)에 맞는 진짜 encrypted inference다 - 유일한 decrypt 지점은 최종
class score(y_hat)뿐이다(client_assisted/inference.py의 "client가 아는 유일한 예외적
decrypt 지점" 원칙과 동일).

test set 전체를 gradient_soft_tree 특유의 SIMD packing(모든 test sample을 한 ciphertext의
슬롯에 packing)으로 한 번에 처리한다 - client_assisted/inference.py의 "sample 하나씩
순차 처리" 방식과 달리, forward 1회 호출로 test sample 전체의 score가 나온다.

python -m models.gradient_soft_tree.packed.predict_packed <session_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from core.data.serialization import load_context  # noqa: E402
from core.ckks_engine import create_bootstrap_engine  # noqa: E402
from models.gradient_soft_tree.packed.block_ops import (  # noqa: E402
    assert_layout_fits,
    build_block_masks,
    compute_block_size,
    pack_dataset_features_blocked,
)
from models.gradient_soft_tree.packed.tree_ops_packed import predict_packed  # noqa: E402


def main() -> None:
    session_dir = Path(sys.argv[1])
    config = json.loads((session_dir / "config.json").read_text())
    n_features = config["n_features"]
    n_classes = config["n_classes"]
    depth = config["depth"]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth

    print(f"[predict_packed] session={session_dir} dataset={config['dataset_name']} depth={depth} - 모델을 decrypt하지 않고 추론합니다", flush=True)

    engine = create_bootstrap_engine(
        mode=config["mode"], device_id=config["device_id"], level_preset=config.get("level_preset")
    )
    ctx = load_context(engine, session_dir / "keys", mode=config["mode"], device_id=config["device_id"])

    params_dir = session_dir / "params"
    params = {
        "alpha": [engine.read_ciphertext(params_dir / f"alpha_{i}.ct") for i in range(n_internal)],
        "threshold": [
            [engine.read_ciphertext(params_dir / f"threshold_{i}_{j}.ct") for j in range(n_features)]
            for i in range(n_internal)
        ],
        "leaf_logits": [engine.read_ciphertext(params_dir / f"leaf_{l}.ct") for l in range(n_leaves)],
    }

    # --- test set을 새로 encrypt (학습 때 쓴 train set과 별개, client가 하는 유일한 encrypt 작업) ---
    X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(config["dataset_name"])
    y_test_oh = one_hot_encode(y_test, n_classes)  # encrypt_dataset 시그니처상 필요(추론엔 안 씀)
    dataset_test = encrypt_dataset(ctx, X_test, y_test_oh)
    sample_mask_test = engine.encrypt([1.0] * dataset_test.n_samples, ctx.pk)

    block_size = compute_block_size(dataset_test.n_samples)  # test set 크기 기준 - threshold는 packed 포맷이 아니라서 학습 때 block_size와 달라도 무관
    assert_layout_fits(n_features, block_size, engine.slot_count)
    block_masks = build_block_masks(n_features, block_size, engine.slot_count)
    blocked_features_test = pack_dataset_features_blocked(ctx, dataset_test.enc_features, block_size)

    print(f"[predict_packed] test set encrypt 완료 (n_test={dataset_test.n_samples}, block_size={block_size}) - forward 실행 중...", flush=True)

    y_hat = predict_packed(
        ctx, dataset_test, params, sample_mask_test, blocked_features_test,
        block_masks, block_size, n_features, n_classes, depth,
    )

    # --- 유일한 decrypt 지점: 최종 class score ---
    n_test = dataset_test.n_samples
    scores = np.array([np.real(engine.decrypt(y_hat[c], ctx.sk))[:n_test] for c in range(n_classes)])  # (n_classes, n_test)
    pred = scores.argmax(axis=0)
    acc = (pred == y_test).mean()

    print(f"[predict_packed] encrypted inference 완료: test_acc={acc:.4f} ({int((pred==y_test).sum())}/{n_test})", flush=True)
    print(f"[predict_packed] 예측: {pred.tolist()}", flush=True)
    print(f"[predict_packed] 정답: {y_test.tolist()}", flush=True)


if __name__ == "__main__":
    main()
