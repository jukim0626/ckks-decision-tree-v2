"""depthN 학습의 최초 setup(key 생성, dataset/초기 파라미터 암호화, session_dir에 직렬화)을
별도 프로세스로 실행한다.

**왜 필요한가(2026-08-26 depth=3 실패로 발견)**: train_depthN_ckks.py의 오케스트레이터가
setup까지 자기 프로세스 안에서 직접 하면, 그 프로세스가 (epoch_worker_N을 하나씩 띄우는 동안)
자기 자신의 GPU context/키를 계속 들고 있게 된다 - epoch_worker_N 프로세스와 **동시에** GPU
메모리를 점유해서, depth=3처럼 초기 파라미터가 많은 경우(alpha 7개+threshold 28개+leaf
8개=43개 ciphertext) 부모+자식 합산 메모리가 24GB를 넘어 OOM이 났다(실측: depth=1(파라미터
8개)은 우연히 버텼지만 depth=3은 epoch 1에서 바로 실패). setup을 별도 프로세스로 끝내고
죽게 하면, 오케스트레이터 자신은 GPU를 전혀 안 잡아서 이 문제가 원천적으로 없어진다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from core.data.serialization import write_dataset, write_keys  # noqa: E402
from core.ckks_engine import create_bootstrap_context  # noqa: E402
from models.gradient_soft_tree.depthN_ckks import init_encrypted_params_N  # noqa: E402


def main() -> None:
    session_dir = Path(sys.argv[1])
    dataset_name = sys.argv[2]
    depth = int(sys.argv[3])
    seed = int(sys.argv[4])
    lr = float(sys.argv[5])
    level_preset = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != "none" else None
    max_train = int(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] != "none" else None

    X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name, max_train=max_train)
    n_features = X_train.shape[1]
    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    y_train_oh = one_hot_encode(y_train, n_classes)
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth

    ctx = create_bootstrap_context(mode="gpu", level_preset=level_preset)
    dataset = encrypt_dataset(ctx, X_train, y_train_oh)
    sample_mask = ctx.engine.encrypt([1.0] * dataset.n_samples, ctx.pk)

    session_dir.mkdir(parents=True, exist_ok=True)
    write_keys(ctx, session_dir / "keys")
    write_dataset(ctx, dataset, session_dir / "dataset")
    ctx.engine.write_ciphertext(sample_mask, session_dir / "sample_mask.ct")

    params = init_encrypted_params_N(ctx, n_features, n_classes, depth, seed=seed, slot_count=ctx.engine.slot_count)
    params_dir = session_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_internal):
        ctx.engine.write_ciphertext(params["alpha"][i], params_dir / f"alpha_{i}.ct")
        for j in range(n_features):
            ctx.engine.write_ciphertext(params["threshold"][i][j], params_dir / f"threshold_{i}_{j}.ct")
    for l in range(n_leaves):
        ctx.engine.write_ciphertext(params["leaf_logits"][l], params_dir / f"leaf_{l}.ct")

    config = {
        "n_features": n_features,
        "n_classes": n_classes,
        "n_samples": dataset.n_samples,
        "depth": depth,
        "mode": ctx.mode,
        "device_id": ctx.device_id,
        "lr": lr,
        "dataset_name": dataset_name,
        "seed": seed,
        "max_train": max_train,
        "level_preset": level_preset,
    }
    (session_dir / "config.json").write_text(json.dumps(config))


if __name__ == "__main__":
    main()
