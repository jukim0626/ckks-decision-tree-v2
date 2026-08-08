"""soft_mgi.py(softmax-blend, sign_bootstrap 없음) vs 기존 SIMD tournament argmin
(fully_encrypted_mgi_simd_argmin, sign_bootstrap 기반) depth=1 stump 비교.

bench_simd_argmin_iris.py를 복제해서 만들었다 - 차이는 그 스크립트가 "고른 candidate가
plaintext와 일치하는지/시간"만 봤다면, 여기는 held-out test accuracy까지 계산해서 soft-MGI가
실제로 쓸만한 정확도를 내는지 확인한다.

평가 방법:
- hard(기존): winner의 threshold/feature 신뢰도를 decrypt해서 어느 candidate가 이겼는지
  찾고(기존 코드 train_fully_encrypted_mgi_stump_simd_precise_score와 동일한 방식 -
  threshold/feature는 protocol상 이미 공개되는 정보), 그 candidate의 (decrypt한) left/right
  class count로 hard rule(x[feature]>threshold) 정확도를 계산.
- soft-MGI: candidate를 하나도 확정하지 않고, blended_gate(test set)와 weights로 가중한
  leaf class count(=blended_leaf_counts, 이번 스크립트 전용 - depth=1이라 recursion에
  안 쓰이므로 soft_mgi.py의 "weighted-MGI 재유도 범위 밖" 제약과 무관)를 조합해서 최종
  class_scores만 decrypt (main.py의 실제 protocol과 같은 수준의 공개).
"""

from __future__ import annotations

import sys
import time

import numpy as np

from client_assisted.candidates import make_public_grid_candidates, make_small_public_threshold_grid
from client_assisted.dataset import load_scaled_dataset_subset, one_hot_encode
from fully_encrypted_mgi_stump import create_bootstrap_context, encrypt_dataset, plaintext_mgi_best_split
from fully_encrypted_mgi_simd_argmin import evaluate_all_candidates_packed, simd_reduce_argmin
from soft_mgi import soft_mgi_weights, blended_gate

DATASET = sys.argv[1] if len(sys.argv) > 1 else "iris"
CANDIDATE_COUNT = 3
HARD_SHARPEN_ITERATIONS = 20
BETA_SWEEP = [("low", 1.0), ("medium", 4.0), ("high", 10.0)]


class CountingEngineProxy:
    """desilofhe.Engine의 메서드 호출 횟수를 세기 위한 검증 전용 wrapper (protocol 일부
    아님). Engine은 C-extension 타입이라 인스턴스에 직접 메서드를 monkeypatch할 수
    없어서("attribute is read-only"), ctx.engine 자리에 이 proxy를 대신 꽂아 넣는다
    (ctx는 평범한 dataclass라 engine 필드 자체는 재할당 가능)."""

    def __init__(self, engine, tracked_method_names: list[str]):
        self._engine = engine
        self.counts = {name: 0 for name in tracked_method_names}
        self._tracked = set(tracked_method_names)

    def __getattr__(self, name):
        attr = getattr(self._engine, name)
        if name not in self._tracked or not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            self.counts[name] += 1
            return attr(*args, **kwargs)

        return wrapped


class CallCounter:
    """ctx.engine을 CountingEngineProxy로 교체하고 counts 딕셔너리를 노출."""

    def __init__(self, ctx, method_names: list[str]):
        proxy = CountingEngineProxy(ctx.engine, method_names)
        ctx.engine = proxy
        self.counts = proxy.counts


def blended_leaf_counts(ctx, weights, packed_counts_per_class: list):
    """weights와 packed_counts_per_class(둘 다 슬롯 i=candidate i, evaluate_all_candidates_packed
    의 aux 포맷)를 슬롯별로 곱한 뒤 SIMD 합 -> sum_i weights[i]*counts_i[c] (전체 슬롯 broadcast).

    depth=1 stump의 최종 leaf 분포를 만들기 위한 것으로, soft_mgi.py가 다루는 범위(soft-MGI
    weight 계산/gate blending) 밖의 "이번 평가 스크립트 전용" 로직이다.
    """
    result = []
    for packed_c in packed_counts_per_class:
        weighted = ctx.engine.multiply(weights, packed_c, ctx.rlk)
        weighted = ctx.engine.intt(weighted)
        result.append(ctx.engine.sum(weighted, ctx.rotation_key))
    return result


def decrypt_scalar(ctx, ct) -> float:
    return float(ctx.engine.decrypt(ct, ctx.sk)[0].real)


def decrypt_vector(ctx, ct, n: int) -> np.ndarray:
    return np.array([v.real for v in ctx.engine.decrypt(ct, ctx.sk)[:n]])


def eval_hard_baseline(candidates, X_train, y_train_one_hot, X_test, y_test, n_classes):
    ctx = create_bootstrap_context(mode="gpu")
    counter = CallCounter(ctx, ["sign_bootstrap", "bootstrap", "merge_bootstrap", "multiply"])
    train_dataset = encrypt_dataset(ctx, X_train, y_train_one_hot)
    normalizer = float(X_train.shape[0] ** 2)

    t0 = time.time()
    packed_score, aux, n_pow2, n_features, n_classes_ = evaluate_all_candidates_packed(
        ctx, train_dataset, candidates, normalizer, verbose=False
    )
    _, winner_aux = simd_reduce_argmin(
        ctx, packed_score, aux, n_pow2, normalizer, HARD_SHARPEN_ITERATIONS, verbose=False
    )
    elapsed = time.time() - t0

    winner_threshold = decrypt_scalar(ctx, winner_aux[0])
    feat_conf = [decrypt_scalar(ctx, fc) for fc in winner_aux[1 : 1 + n_features]]
    winner_feature_idx = max(range(n_features), key=lambda i: feat_conf[i])

    best_idx, best_dist = None, None
    for idx, c in enumerate(candidates):
        if c.feature_idx != winner_feature_idx:
            continue
        dist = abs(c.threshold - winner_threshold)
        if best_dist is None or dist < best_dist:
            best_dist, best_idx = dist, idx
    matched = candidates[best_idx]

    left_counts = np.array(
        [decrypt_scalar(ctx, c) for c in winner_aux[1 + n_features : 1 + n_features + n_classes]]
    )
    right_counts = np.array(
        [decrypt_scalar(ctx, c) for c in winner_aux[1 + n_features + n_classes :]]
    )
    pred_left, pred_right = int(np.argmax(left_counts)), int(np.argmax(right_counts))

    preds = [
        pred_right if X_test[i, matched.feature_idx] > matched.threshold else pred_left
        for i in range(X_test.shape[0])
    ]
    accuracy = float(np.mean(np.array(preds) == y_test))

    return {
        "elapsed": elapsed,
        "accuracy": accuracy,
        "matched_candidate": matched,
        "counts": dict(counter.counts),
    }


def eval_soft_mgi(candidates, X_train, y_train_one_hot, X_test, y_test, n_classes, beta):
    ctx = create_bootstrap_context(mode="gpu")
    counter = CallCounter(ctx, ["sign_bootstrap", "bootstrap", "merge_bootstrap", "multiply"])
    train_dataset = encrypt_dataset(ctx, X_train, y_train_one_hot)
    test_dataset = encrypt_dataset(ctx, X_test, one_hot_encode(y_test, n_classes=n_classes))
    normalizer = float(X_train.shape[0] ** 2)
    n_valid = len(candidates)

    t0 = time.time()
    packed_score, aux, n_pow2, n_features, n_classes_ = evaluate_all_candidates_packed(
        ctx, train_dataset, candidates, normalizer, verbose=False
    )
    weights = soft_mgi_weights(ctx, packed_score, n_valid, n_pow2, normalizer, beta)
    gate = blended_gate(ctx, test_dataset.enc_features, candidates, weights)

    packed_left = aux[1 + n_features : 1 + n_features + n_classes]
    packed_right = aux[1 + n_features + n_classes :]
    leaf_left = blended_leaf_counts(ctx, weights, packed_left)
    leaf_right = blended_leaf_counts(ctx, weights, packed_right)
    elapsed = time.time() - t0

    n_test = X_test.shape[0]
    gate_vals = decrypt_vector(ctx, gate, n_test)
    leaf_left_vals = np.array([decrypt_scalar(ctx, c) for c in leaf_left])
    leaf_right_vals = np.array([decrypt_scalar(ctx, c) for c in leaf_right])

    preds = []
    for j in range(n_test):
        g = gate_vals[j]
        scores = g * leaf_right_vals + (1.0 - g) * leaf_left_vals
        preds.append(int(np.argmax(scores)))
    accuracy = float(np.mean(np.array(preds) == y_test))

    return {
        "elapsed": elapsed,
        "accuracy": accuracy,
        "counts": dict(counter.counts),
    }


def main() -> None:
    X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(
        dataset_name=DATASET, test_size=30
    )
    n_classes = len(class_names)
    y_train_one_hot = one_hot_encode(y_train, n_classes=n_classes)
    candidates = make_public_grid_candidates(
        n_features=X_train.shape[1],
        thresholds=make_small_public_threshold_grid(candidate_count=CANDIDATE_COUNT),
    )
    print(
        f"[setup] dataset={DATASET} candidates={len(candidates)} "
        f"train={X_train.shape[0]} test={X_test.shape[0]}",
        flush=True,
    )
    best_idx, best_score = plaintext_mgi_best_split(X_train, y_train_one_hot, candidates)
    print(f"[plaintext MGI] best={candidates[best_idx]} score={best_score:.4f}\n", flush=True)

    print(f"=== hard (SIMD tournament argmin, sharpen={HARD_SHARPEN_ITERATIONS}) ===", flush=True)
    hard = eval_hard_baseline(candidates, X_train, y_train_one_hot, X_test, y_test, n_classes)
    print(
        f"  time={hard['elapsed']:.2f}s accuracy={hard['accuracy']:.4f} "
        f"matched={hard['matched_candidate']}"
    )
    print(f"  call counts={hard['counts']}\n", flush=True)

    results = []
    for label, beta in BETA_SWEEP:
        print(f"=== soft-MGI (beta={beta}, {label}) ===", flush=True)
        soft = eval_soft_mgi(candidates, X_train, y_train_one_hot, X_test, y_test, n_classes, beta)
        print(f"  time={soft['elapsed']:.2f}s accuracy={soft['accuracy']:.4f}")
        print(f"  call counts={soft['counts']}\n", flush=True)
        results.append((label, beta, soft))

    print("=== summary ===")
    print(f"{'variant':<16}{'accuracy':>10}{'time(s)':>10}{'sign_bootstrap calls':>22}{'multiply calls':>16}")
    print(
        f"{'hard(tournament)':<16}{hard['accuracy']:>10.4f}{hard['elapsed']:>10.2f}"
        f"{hard['counts']['sign_bootstrap']:>22}{hard['counts']['multiply']:>16}"
    )
    for label, beta, soft in results:
        name = f"soft(beta={beta:g},{label})"
        print(
            f"{name:<16}{soft['accuracy']:>10.4f}{soft['elapsed']:>10.2f}"
            f"{soft['counts']['sign_bootstrap']:>22}{soft['counts']['multiply']:>16}"
        )


if __name__ == "__main__":
    main()
