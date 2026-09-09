"""depth1_ckks.py의 forward_backward_update를 epoch마다 별도 프로세스(epoch_worker.py)로
실행하는 오케스트레이터. desilofhe GPU 메모리 누적 문제(EXPERIMENT_LOG.md 2026-08-11/12) 때문에
단일 프로세스로는 epoch 3에서 CUDA OOM이 났던 걸(depth1_ckks.py 단독 실행 실측) node_worker.py/
train.py와 같은 패턴(session_dir에 key/dataset/params를 직렬화, 매 epoch마다 subprocess 실행)
으로 우회한다.

사용법: python -m experiments.gradient_soft_tree.train_depth1_ckks iris 20
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from client_assisted.dataset import encrypt_dataset, load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from closed_form_mgi.io_utils import write_dataset, write_keys  # noqa: E402
from closed_form_mgi.primitives import create_bootstrap_context  # noqa: E402
from experiments.gradient_soft_tree.depth1_ckks import STEEPNESS, decrypt_params, init_encrypted_params  # noqa: E402
from experiments.gradient_soft_tree.depth1_reference import (  # noqa: E402
    predict as plaintext_predict,
    train_depth1 as plaintext_train_depth1,
)


def _run_epoch_worker(session_dir: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "experiments.gradient_soft_tree.epoch_worker", str(session_dir)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"epoch_worker 실패:\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


def main() -> None:
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "iris"
    n_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    lr = 0.3
    seed = 0

    X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name)
    n_features = X_train.shape[1]
    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    y_train_oh = one_hot_encode(y_train, n_classes)

    print(f"[setup] dataset={dataset_name} n_samples={X_train.shape[0]} n_features={n_features} n_classes={n_classes} epochs={n_epochs}")
    ctx = create_bootstrap_context(mode="gpu")
    dataset = encrypt_dataset(ctx, X_train, y_train_oh)
    sample_mask = ctx.engine.encrypt([1.0] * dataset.n_samples, ctx.pk)

    session_dir = Path(tempfile.mkdtemp(prefix="gradient_soft_tree_depth1_"))
    write_keys(ctx, session_dir / "keys")
    write_dataset(ctx, dataset, session_dir / "dataset")
    ctx.engine.write_ciphertext(sample_mask, session_dir / "sample_mask.ct")

    params = init_encrypted_params(ctx, n_features, n_classes, seed=seed, slot_count=ctx.engine.slot_count)
    params.pop("plaintext_init")
    params_dir = session_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    ctx.engine.write_ciphertext(params["alpha"], params_dir / "alpha.ct")
    for j, t in enumerate(params["threshold"]):
        ctx.engine.write_ciphertext(t, params_dir / f"threshold_{j}.ct")
    ctx.engine.write_ciphertext(params["leaf_L"], params_dir / "leaf_L.ct")
    ctx.engine.write_ciphertext(params["leaf_R"], params_dir / "leaf_R.ct")

    config = {
        "n_features": n_features,
        "n_classes": n_classes,
        "n_samples": dataset.n_samples,
        "mode": ctx.mode,
        "device_id": ctx.device_id,
        "lr": lr,
    }
    (session_dir / "config.json").write_text(json.dumps(config))

    print("[reference] plaintext trajectory (동일 seed) 계산 중...")
    ref_final = plaintext_train_depth1(X_train, y_train_oh, steepness=STEEPNESS, lr=lr, epochs=n_epochs, seed=seed)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        _run_epoch_worker(session_dir)
        elapsed = time.time() - t0
        print(f"[epoch {epoch}/{n_epochs}] done | {elapsed:.1f}s", flush=True)

    # **검증/평가 전용 decrypt** (protocol 일부 아님 - closed_form_mgi의 debug_decrypt_winner와
    # 같은 지위). 학습 자체는 위 루프에서 전부 ciphertext로만 진행됐다.
    engine = ctx.engine
    final_params = {
        "alpha": engine.read_ciphertext(params_dir / "alpha.ct"),
        "threshold": [engine.read_ciphertext(params_dir / f"threshold_{j}.ct") for j in range(n_features)],
        "leaf_L": engine.read_ciphertext(params_dir / "leaf_L.ct"),
        "leaf_R": engine.read_ciphertext(params_dir / "leaf_R.ct"),
    }
    decoded = decrypt_params(ctx, final_params, n_features, n_classes)
    max_err = max(
        np.abs(decoded["alpha"] - ref_final["alpha"]).max(),
        np.abs(decoded["threshold"] - ref_final["threshold"]).max(),
        np.abs(decoded["leaf_L"] - ref_final["leaf_L"]).max(),
        np.abs(decoded["leaf_R"] - ref_final["leaf_R"]).max(),
    )
    print(f"\n[final] max abs diff vs plaintext reference (after {n_epochs} epochs) = {max_err:.5f}")

    train_pred = plaintext_predict(X_train, decoded)
    test_pred = plaintext_predict(X_test, decoded)
    train_acc = (train_pred == y_train).mean()
    test_acc = (test_pred == y_test).mean()
    print(f"[final] CKKS로 학습된 파라미터 -> train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
    print(f"[session_dir] {session_dir}")


if __name__ == "__main__":
    main()
