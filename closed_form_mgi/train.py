"""closed-form threshold(grid 없음) + soft-MGI blend 학습 (CKKS). depth>=1 재귀 지원.

2026-08-06 plaintext 검증(closed_form_mgi/debug/bench_split_methods_plaintext.py)에서
public grid(feature*3 candidates)를 없애고 feature마다 그 노드의 (node_weight로 가중한)
평균값을 threshold로 직접 계산하는 게, 정확도는 비슷하거나 더 좋으면서 candidate 수가
1/3로 줄어(feature*3 -> feature) 비용도 1/3 수준으로 낮다는 걸 확인했다. 이 파일은 그
방식을 실제 CKKS로 포팅한 것 (학습 전용 - 추론/라우팅은 closed_form_mgi/inference.py).

**아직 안 푼 부분**: plaintext 검증에서 depth>=2는 softmax 정규화를 "그 노드 K개 후보
점수의 min/max 범위"로 해야 beta<=50 안에서 hard-tournament 수준 정확도가 나왔는데(고정
정규화 n_samples**2로는 안 됨), encrypted min/max(비교 없이)는 아직 못 만들었다
(LogSumExp 기반 soft-min을 plaintext로 검증은 했지만 log() 다항식 근사가 새 primitive라
CKKS로 아직 안 옮겼다). 이 파일은 우선 **고정 정규화(score_normalizer=n_samples**2)**만
쓴다 - depth=1은 이걸로 충분(고정/range 정규화 정확도 동일 확인됨), depth>=2는 정확도가
plaintext range-normalization 버전보다 낮을 것으로 예상됨(알려진 한계, 실측 필요).

**노드별 프로세스 분리 (2026-08-11)**: 노드 하나 처리(threshold의 30회 + soft-MGI weight의
40회 뉴턴-랩슨 반복)를 거치면 GPU 메모리가 계속 쌓이는데, ciphertext를 개별적으로
del+gc.collect()해도 거의 안 줄어든다 - desilofhe가 key가 살아있는 동안 ciphertext 메모리
풀 전체를 붙잡고 있어서, key까지 지워야(=프로세스가 죽어야) OS가 회수한다(leak_probe 실측:
del+gc.collect(loop 변수만)로는 9404->9380MiB, key까지 지우면 9380->5288MiB). 그래서
`train_closed_form_mgi_tree`는 노드 계산 자체를 하지 않고, key/dataset을 임시 디렉터리에
직렬화해둔 뒤 노드마다 `node_worker.py`를 별도 프로세스로 실행해서 그 프로세스가 끝날 때
GPU 메모리가 강제로 회수되게 한다 (`ensure_level(min_level=16)` 파라미터를 줄여서 ciphertext
자체를 작게 만드는 시도는 production beta=30에서 정확도가 깨져서 기각 - EXPERIMENT_LOG.md
2026-08-11 참고).
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ckks_tree import sigmoid_approx_enc
from closed_form_mgi.io_utils import write_dataset, write_keys
from closed_form_mgi.model import ClosedFormMgiNode, ClosedFormMgiTreeModel
from closed_form_mgi.primitives import (
    encrypted_mgi_from_counts,
    encrypted_weighted_class_counts,
    ensure_level,
)
from closed_form_mgi.simd_argmin import next_power_of_two, scatter_to_slot
from closed_form_mgi.soft_mgi import _extract_weight_broadcast, soft_mgi_weights

MEAN_RECIPROCAL_ITERATIONS = 30


def encrypted_weighted_threshold(ctx, enc_feature, enc_node_weights, n_samples: int, iterations: int = MEAN_RECIPROCAL_ITERATIONS):
    """threshold = weighted_mean(feature) = sum(weight*x)/sum(weight).

    soft_mgi.soft_mgi_weights와 같은 bounded Newton-Raphson 패턴: y0=1/n_samples(공개
    상한 기반 안전 초기값), z=weighted_count*y(0~1로 수렴, bootstrap-safe), w=weighted_sum*y
    (feature 값이 [-1,1] 안이라 가중평균도 자연히 [-1,1] 안 - bootstrap-safe). 수렴하면
    w == 원하는 threshold.
    """
    weighted_feature = ctx.engine.multiply(enc_feature, enc_node_weights, ctx.rlk)
    weighted_feature = ctx.engine.intt(weighted_feature)
    weighted_sum = ctx.engine.sum(weighted_feature, ctx.rotation_key)

    node_weights_for_sum = ctx.engine.intt(enc_node_weights)
    weighted_count = ctx.engine.sum(node_weights_for_sum, ctx.rotation_key)

    y0 = 1.0 / n_samples
    z = ctx.engine.multiply(weighted_count, y0)
    w = ctx.engine.multiply(weighted_sum, y0)
    for i in range(iterations):
        z = ensure_level(ctx, z)
        w = ensure_level(ctx, w)
        two_minus_z = ctx.engine.subtract(2.0, z)
        z_new = ctx.engine.multiply(z, two_minus_z, ctx.rlk)
        w = ctx.engine.multiply(w, two_minus_z, ctx.rlk)
        z = z_new
        del two_minus_z
        if i % 5 == 0:
            gc.collect()
    return w


def blended_gate_from_gates(ctx, gates: list, weights):
    """feature별 gate(각 gate는 "오른쪽으로 갈 확률")를 soft-MGI weight로 가중합.
    학습 중(자기 자신의 train weight 전파)과 추론 라우팅(inference.py, 새 dataset에 적용) 둘
    다에서 재사용되는 공용 함수라 여기(train.py)에 둔다."""
    weights = ensure_level(ctx, weights, min_level=14)
    gate = None
    for i, g_i in enumerate(gates):
        g_i = ensure_level(ctx, g_i, min_level=14)
        w_i = _extract_weight_broadcast(ctx, weights, i)
        piece = ctx.engine.multiply(w_i, g_i, ctx.rlk)
        gate = piece if gate is None else ctx.engine.add(gate, piece)
    return ensure_level(ctx, gate, min_level=16)


def evaluate_closed_form_candidates_from_thresholds(ctx, dataset, enc_node_weights, thresholds: list, score_normalizer: float):
    """미리 계산된 threshold(feature별)로 gate + MGI 점수를 계산해 SIMD 슬롯에 packing."""
    n_features = dataset.n_features
    n_pow2 = next_power_of_two(n_features)
    gates = []
    packed_score = None
    for j in range(n_features):
        enc_feature = ensure_level(ctx, dataset.enc_features[j])
        threshold = ensure_level(ctx, thresholds[j])
        enc_diff = ctx.engine.subtract(enc_feature, threshold)
        gate = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff)
        gates.append(gate)

        weighted_gate = ctx.engine.multiply(gate, enc_node_weights, ctx.rlk)
        left_prob = ctx.engine.subtract(1.0, gate)
        weighted_left = ctx.engine.multiply(left_prob, enc_node_weights, ctx.rlk)
        left_counts = encrypted_weighted_class_counts(ctx, weighted_left, dataset.enc_labels)
        right_counts = encrypted_weighted_class_counts(ctx, weighted_gate, dataset.enc_labels)
        score = ctx.engine.add(
            encrypted_mgi_from_counts(ctx, left_counts),
            encrypted_mgi_from_counts(ctx, right_counts),
        )
        # score는 최대 n_samples**2(~수천~수만) 크기라 bootstrap-safe 범위(~2~5)를 훨씬
        # 넘는다. ensure_level(=필요시 bootstrap)을 걸기 *전에* 반드시 score_normalizer로
        # 나눠서 [0,1] 안으로 넣어야 한다 (soft_mgi.py의 "값 크기 2~5 넘으면 bootstrap이
        # 조용히 깨진다" 버그와 동일 원인).
        score = ctx.engine.multiply(score, 1.0 / score_normalizer)
        score = ensure_level(ctx, score, min_level=14)
        piece = scatter_to_slot(ctx, score, j)
        packed_score = piece if packed_score is None else ctx.engine.add(packed_score, piece)
        gc.collect()

    if n_pow2 > n_features:
        pad_mask = ctx.engine.encrypt([0.0] * n_features + [1.0] * (n_pow2 - n_features), ctx.pk)
        sentinel = ctx.engine.encrypt([1.0] * n_pow2, ctx.pk)  # 이미 정규화된 스케일이라 "최악"=1.0
        packed_score = ctx.engine.add(packed_score, ctx.engine.multiply(pad_mask, sentinel, ctx.rlk))

    packed_score = ensure_level(ctx, packed_score, min_level=16)
    return packed_score, gates, n_pow2, n_features


def process_single_node(ctx, dataset, enc_train_weights, beta: float, score_normalizer: float):
    """노드 하나의 threshold/soft-MGI weight/blended gate 계산 - node_worker.py(별도
    프로세스)와 이 파일 양쪽에서 재사용하는 핵심 로직. 반환: (thresholds, weights,
    left_train, right_train)."""
    enc_train_weights = ensure_level(ctx, enc_train_weights, min_level=16)

    thresholds = [
        encrypted_weighted_threshold(ctx, dataset.enc_features[j], enc_train_weights, dataset.n_samples)
        for j in range(dataset.n_features)
    ]
    packed_score, gates, n_pow2, n_features = evaluate_closed_form_candidates_from_thresholds(
        ctx, dataset, enc_train_weights, thresholds, score_normalizer
    )
    # packed_score는 이미 evaluate_closed_form_candidates_from_thresholds 안에서
    # score_normalizer로 정규화됐으므로, 여기서는 1.0을 넘겨 이중 정규화를 막는다.
    weights = soft_mgi_weights(ctx, packed_score, n_features, n_pow2, 1.0, beta)
    blended_gate = blended_gate_from_gates(ctx, gates, weights)

    right_train = ctx.engine.multiply(enc_train_weights, blended_gate, ctx.rlk)
    left_train = ctx.engine.multiply(enc_train_weights, ctx.engine.subtract(1.0, blended_gate), ctx.rlk)
    return thresholds, weights, left_train, right_train


def _run_worker(session_dir: Path, module: str, *args: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, str(session_dir), *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{module} 실패 (args={args}):\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


def train_closed_form_mgi_tree(
    ctx,
    train_dataset,
    depth: int,
    beta: float = 30.0,
    score_normalizer: float | None = None,
    verbose: bool = True,
) -> ClosedFormMgiTreeModel:
    """client decrypt 없이, grid 없는 closed-form threshold + soft-MGI blend로 depth짜리
    tree를 학습. 반환: ClosedFormMgiTreeModel (노드별 thresholds/weights + leaf_counts,
    전부 pre-order). 새 dataset(test든 임의 샘플이든)을 이 모델로 라우팅하려면
    closed_form_mgi/inference.py의 route_dataset_through_model()을 쓴다.

    **노드마다 별도 프로세스**(node_worker.py)에서 계산한다 - 2026-08-11 실측: desilofhe는
    key가 살아있는 동안 ciphertext 메모리 풀을 계속 붙잡고 있어서(del+gc.collect()로는 GPU
    메모리가 거의 안 줄고, key까지 지워야 줄어듦), 트리 전체를 한 프로세스 안에서 학습하면
    depth>=2에서 CUDA OOM이 났다(파라미터를 줄이는 시도는 production beta=30에서 정확도가
    깨져서 기각 - EXPERIMENT_LOG.md 참고). key/dataset은 세션 디렉터리에 한 번만 직렬화해두고
    노드마다 새 프로세스를 띄워 읽게 해서, 프로세스가 끝날 때 OS가 GPU 메모리를 강제로
    회수하게 만든다."""
    if score_normalizer is None:
        score_normalizer = float(train_dataset.n_samples**2)
    if depth < 1:
        raise ValueError("depth must be at least 1")

    total_nodes = (1 << depth) - 1
    session_dir = Path(tempfile.mkdtemp(prefix="closed_form_mgi_"))
    write_keys(ctx, session_dir / "keys")
    write_dataset(ctx, train_dataset, session_dir / "dataset")
    config = {
        "n_features": train_dataset.n_features,
        "n_classes": train_dataset.n_classes,
        "n_samples": train_dataset.n_samples,
        "beta": beta,
        "score_normalizer": score_normalizer,
        "mode": ctx.mode,
        "device_id": ctx.device_id,
    }
    (session_dir / "config.json").write_text(json.dumps(config))

    (session_dir / "nodes" / "root").mkdir(parents=True, exist_ok=True)
    root_weights = ctx.engine.encrypt([1.0] * train_dataset.n_samples, ctx.pk)
    ctx.engine.write_ciphertext(root_weights, session_dir / "nodes" / "root" / "input_weights.ct")
    del root_weights

    nodes: list = []
    leaf_ids: list = []
    node_count = 0
    # (node_id, depth) 스택 - 오른쪽을 먼저 push해서 왼쪽이 먼저 pop되게 하면 기존
    # 재귀(train_node(left,...) 먼저 호출)와 동일한 pre-order가 유지된다. node_id는
    # root에서부터의 L/R 경로 문자열이라 그대로 디렉터리 이름으로 쓴다.
    stack = [("root", 0)]
    while stack:
        node_id, current_depth = stack.pop()
        if current_depth == depth:
            leaf_ids.append(node_id)
            continue

        t0 = time.time()
        _run_worker(session_dir, "closed_form_mgi.node_worker", node_id)
        node_dir = session_dir / "nodes" / node_id

        # 결과를 여기서 바로 ctx.engine.read_ciphertext()로 읽지 않는다 - 2026-08-11 실측:
        # key가 살아있는 부모 프로세스가 노드마다 결과를 즉시 읽어들이면(fresh read든
        # to_cuda든 상관없이) 그 메모리 풀이 계속 붙잡혀서 부모 자체가 다시 누적 OOM에
        # 걸렸다. 파일 경로만 모델에 저장해두고, inference.py가 실제로 그 노드를 쓸 때만
        # 그때그때 읽고 버리도록 미룬다 (ClosedFormMgiNode 참고).
        thresholds = [node_dir / f"threshold_{j}.ct" for j in range(train_dataset.n_features)]
        weights = node_dir / "weights.ct"
        nodes.append(ClosedFormMgiNode(thresholds=thresholds, weights=weights))

        left_id, right_id = node_id + "L", node_id + "R"
        (session_dir / "nodes" / left_id).mkdir(parents=True, exist_ok=True)
        (session_dir / "nodes" / right_id).mkdir(parents=True, exist_ok=True)
        (node_dir / "left_weights.ct").rename(session_dir / "nodes" / left_id / "input_weights.ct")
        (node_dir / "right_weights.ct").rename(session_dir / "nodes" / right_id / "input_weights.ct")

        node_count += 1
        elapsed = time.time() - t0
        if verbose:
            print(f"[closed-form train] node {node_count}/{total_nodes} (depth {current_depth}) done | {elapsed:.1f}s", flush=True)

        stack.append((right_id, current_depth + 1))
        stack.append((left_id, current_depth + 1))

    _run_worker(session_dir, "closed_form_mgi.leaf_worker", ",".join(leaf_ids))
    leaf_counts = [
        [session_dir / "nodes" / leaf_id / f"count_{c}.ct" for c in range(train_dataset.n_classes)]
        for leaf_id in leaf_ids
    ]

    return ClosedFormMgiTreeModel(depth=depth, nodes=nodes, leaf_counts=leaf_counts, session_dir=session_dir)
