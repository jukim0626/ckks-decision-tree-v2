"""local_loss 학습의 최초 setup(key 생성, dataset/초기 파라미터 암호화, session_dir에 직렬화)을
별도 프로세스로 실행한다. baseline/setup_worker.py와 완전히 같은 목적/구조 - GPU를 만지는
모든 작업(setup/epoch/finalize)을 각자 별도 프로세스로 분리해서, desilofhe가 한 프로세스
안에서 GPU 메모리를 절대 안 돌려주는 문제를 프로세스 종료로 우회한다.

**local_loss 전용 차이**: baseline은 leaf_logits가 리스트 하나(길이 2^depth)인데, local_loss는
레벨마다 자기만의 local classifier가 있어서 `local_logits`가 **레벨별로 길이가 다른 리스트의
리스트**(level 0은 길이 2, level 1은 길이 4, ...)다. 파일명에 level/index를 둘 다 인코딩해서
직렬화한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from core.data.serialization import write_dataset, write_keys  # noqa: E402
from core.ckks_engine import create_bootstrap_context  # noqa: E402
from models.gradient_soft_tree.local_loss.tree_ops import init_encrypted_params_N  # noqa: E402


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
    for level in range(depth):
        for k in range(1 << (level + 1)):
            ctx.engine.write_ciphertext(params["local_logits"][level][k], params_dir / f"local_{level}_{k}.ct")

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
