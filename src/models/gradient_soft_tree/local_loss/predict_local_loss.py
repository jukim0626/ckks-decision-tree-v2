"""학습된(암호화 상태 그대로 저장된) session_dir을 읽어서, **모델을 한 번도 decrypt하지
않고** test set에 대해 진짜 encrypted inference를 수행한다. `finalize_worker.py`가 지금까지
계산해온 train_acc/test_acc는 사실 "params를 decrypt한 뒤 평문으로 predict"한 것(검증용)이었고,
이 스크립트가 이 프로젝트의 목표("학습과 추론 모두를 암호화 상태로 수행", CLAUDE.md)에 맞는
진짜 encrypted inference다 - 유일한 decrypt 지점은 최종 class score(y_hat)뿐이다
(packed/predict_packed.py와 동일한 원칙).

forward 구조는 baseline과 100% 동일(gate.py 공유, `compute_axis_aligned_gate`) - local_loss가
학습 때 레벨마다 local classifier를 따로 뒀더라도, 추론에서는 **가장 깊은 레벨(depth-1)의
local classifier(=leaf_logits)만** 쓴다(reference.py의 predict()가 predict_baseline을 그대로
재사용하는 것과 같은 이유 - depth-1 레벨의 가상 leaf가 곧 진짜 leaf).

python -m models.gradient_soft_tree.local_loss.predict_local_loss <session_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import encrypt_dataset, load_scaler, one_hot_encode, split_dataset_subset  # noqa: E402
from core.data.serialization import load_context  # noqa: E402
from core.ckks_engine import create_bootstrap_engine, ensure_level  # noqa: E402
from core.encrypted_ops.slot_packing import extract_weight_broadcast, next_power_of_two  # noqa: E402
from core.encrypted_ops.softmax import packed_softmax  # noqa: E402
from models.gradient_soft_tree.gate import compute_axis_aligned_gate  # noqa: E402

_MIN_LEVEL = 5


def predict_local_loss(ctx, dataset, params: dict, n_features: int, n_classes: int, depth: int):
    """params={"alpha","threshold","leaf_logits"(=depth-1 레벨의 local_logits, 길이 2^depth)}.
    forward는 tree_ops.forward_backward_update_N의 forward 절반과 동일, backward/얕은 레벨의
    local classifier는 추론에 안 쓰이므로 생략."""
    n_pow2_f = next_power_of_two(n_features)
    n_pow2_c = next_power_of_two(n_classes)

    current_level_probs = [None]
    node_idx = 0
    for _level in range(depth):
        count = 1 << _level
        next_level_probs = []
        for _idx, parent_prob in enumerate(current_level_probs):
            i = node_idx
            gate, _gate_terms, _w = compute_axis_aligned_gate(
                ctx, dataset, params["alpha"][i], params["threshold"][i], n_features, n_pow2_f, min_level=_MIN_LEVEL
            )
            if parent_prob is None:
                left = ctx.engine.subtract(1.0, gate)
                right = gate
            else:
                parent_prob = ensure_level(ctx, parent_prob, min_level=_MIN_LEVEL)
                left = ctx.engine.multiply(parent_prob, ctx.engine.subtract(1.0, gate), ctx.rlk)
                right = ctx.engine.multiply(parent_prob, gate, ctx.rlk)
            next_level_probs.append(ensure_level(ctx, left, min_level=_MIN_LEVEL))
            next_level_probs.append(ensure_level(ctx, right, min_level=_MIN_LEVEL))
            node_idx += 1
        current_level_probs = next_level_probs

    n_leaves = 1 << depth
    local_dist = [packed_softmax(ctx, params["leaf_logits"][k], n_classes, n_pow2_c) for k in range(n_leaves)]

    y_hat = []
    for c in range(n_classes):
        acc = None
        for k in range(n_leaves):
            ld_c = ensure_level(ctx, extract_weight_broadcast(ctx, local_dist[k], c), min_level=_MIN_LEVEL)
            term = ctx.engine.multiply(current_level_probs[k], ld_c, ctx.rlk)
            acc = term if acc is None else ctx.engine.add(acc, term)
        y_hat.append(ensure_level(ctx, acc, min_level=_MIN_LEVEL))
    return y_hat


def main() -> None:
    session_dir = Path(sys.argv[1])
    config = json.loads((session_dir / "config.json").read_text())
    n_features = config["n_features"]
    n_classes = config["n_classes"]
    depth = config["depth"]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth

    print(f"[predict_local_loss] session={session_dir} dataset={config['dataset_name']} depth={depth} - 모델을 decrypt하지 않고 추론합니다", flush=True)

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
        "leaf_logits": [engine.read_ciphertext(params_dir / f"local_{depth - 1}_{k}.ct") for k in range(n_leaves)],
    }

    # --- test set을 새로 encrypt (학습 때 쓴 train set과 별개, client가 하는 유일한 encrypt 작업) ---
    # 2026-09-21: 예전엔 max_train을 안 넘겨서(local_loss_packed의 breast_cancer 세션처럼
    # max_train을 실제로 쓴 경우) 학습 때와 다른 train subsample로 scaler가 fit돼 평가가
    # 어긋날 위험이 있었다 - config에 저장된 실제 값으로 raw split을 재현하고, 학습 때
    # 저장해둔 scaler로 transform만 한다(재적합 없음).
    X_train_raw, X_test_raw, y_train, y_test, _ = split_dataset_subset(
        config["dataset_name"], test_size=config.get("test_size", 0.2), max_train=config.get("max_train")
    )
    scaler = load_scaler(
        session_dir / "client" / "scaler.json",
        expected_dataset_name=config["dataset_name"],
        expected_n_features=n_features,
    )
    X_test = scaler.transform(X_test_raw)
    y_test_oh = one_hot_encode(y_test, n_classes)  # encrypt_dataset 시그니처상 필요(추론엔 안 씀)
    dataset_test = encrypt_dataset(ctx, X_test, y_test_oh)

    print(f"[predict_local_loss] test set encrypt 완료 (n_test={dataset_test.n_samples}) - forward 실행 중...", flush=True)

    y_hat = predict_local_loss(ctx, dataset_test, params, n_features, n_classes, depth)

    # --- 유일한 decrypt 지점: 최종 class score ---
    n_test = dataset_test.n_samples
    scores = np.array([np.real(engine.decrypt(y_hat[c], ctx.sk))[:n_test] for c in range(n_classes)])  # (n_classes, n_test)
    pred = scores.argmax(axis=0)
    acc = (pred == y_test).mean()

    print(f"[predict_local_loss] encrypted inference 완료: test_acc={acc:.4f} ({int((pred==y_test).sum())}/{n_test})", flush=True)
    print(f"[predict_local_loss] 예측: {pred.tolist()}", flush=True)
    print(f"[predict_local_loss] 정답: {y_test.tolist()}", flush=True)


if __name__ == "__main__":
    main()
