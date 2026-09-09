"""setup_worker_N.py(axis-aligned)와 완전히 같은 구조 - 파라미터가 alpha/threshold ->
w/b로 바뀐 것만 다르다. 오케스트레이터 자신이 GPU를 안 잡도록 별도 프로세스로 분리하는
이유도 동일(depthN_ckks.py 상단 주석 참고)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from client_assisted.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from closed_form_mgi.io_utils import write_dataset, write_keys  # noqa: E402
from closed_form_mgi.primitives import create_bootstrap_context  # noqa: E402
from experiments.gradient_soft_tree.oblique.depthN_ckks import init_encrypted_params_N  # noqa: E402


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
        for j in range(n_features):
            ctx.engine.write_ciphertext(params["w"][i][j], params_dir / f"w_{i}_{j}.ct")
        ctx.engine.write_ciphertext(params["b"][i], params_dir / f"b_{i}.ct")
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
