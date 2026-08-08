"""node5(root의 오른쪽 자식, plaintext gap=0.0118로 "쉬워야 할" 노드)의 실제 SIMD
reduction을 라운드별로 decrypt해서, 어느 시점에 confidence가 깨지는지 추적.

root를 실제로 계산해서 진짜 right_weights를 얻은 뒤, 그걸로 node5의 candidate scoring +
argmin reduction을 돌리면서 매 라운드 score 값을 decrypt해서 출력한다.
"""

from __future__ import annotations

import time

import numpy as np

from client_assisted.candidates import make_public_grid_candidates, make_small_public_threshold_grid
from client_assisted.dataset import load_scaled_dataset_subset, one_hot_encode
from client_assisted.server_ops import server_compute_weighted_child_weights
from fully_encrypted_mgi_stump import create_bootstrap_context, encrypt_dataset, ensure_level
from fully_encrypted_mgi_tree import evaluate_all_candidates_packed_weighted, find_best_split_weighted
from fully_encrypted_mgi_simd_argmin import batch_ensure_level, encrypted_blend, sharpened_sign

DATASET = "iris"
CANDIDATE_COUNT = 3
ROOT_SHARPEN = 19
NODE5_SHARPEN = 12  # 지난 실행에서 이 노드에 실제로 쓴 값

X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(dataset_name=DATASET, test_size=30)
y_train_one_hot = one_hot_encode(y_train, n_classes=len(class_names))
candidates = make_public_grid_candidates(
    n_features=X_train.shape[1], thresholds=make_small_public_threshold_grid(candidate_count=CANDIDATE_COUNT)
)
normalizer = float(X_train.shape[0] ** 2)

ctx = create_bootstrap_context(mode="gpu")
dataset = encrypt_dataset(ctx, X_train, y_train_one_hot)
root_weights = ctx.engine.encrypt([1.0] * dataset.n_samples, ctx.pk)

t0 = time.time()
root_split = find_best_split_weighted(ctx, dataset, candidates, root_weights, normalizer, ROOT_SHARPEN, verbose=False)
print(f"[root] done {time.time()-t0:.1f}s", flush=True)

_, right_weights = server_compute_weighted_child_weights(ctx, dataset, root_split, root_weights)
right_weights = ensure_level(ctx, right_weights, min_level=16)

print("\n=== node5(오른쪽 자식) scoring 시작 ===", flush=True)
t0 = time.time()
packed_score, aux, n_pow2, n_features, n_classes = evaluate_all_candidates_packed_weighted(
    ctx, dataset, candidates, right_weights, normalizer, verbose=True
)
print(f"[scoring] done {time.time()-t0:.1f}s", flush=True)

dec = ctx.engine.decrypt(packed_score, ctx.sk)[:n_pow2]
print(f"\nround0(scoring 직후) packed_score(12개 real + padding)={[round(v.real,2) for v in dec]}")

print("\n=== round별 reduction 추적 (NODE5_SHARPEN={}) ===".format(NODE5_SHARPEN))
score = packed_score
n_cur = n_pow2
round_idx = 0
while n_cur > 1:
    half = n_cur // 2
    refreshed = batch_ensure_level(ctx, [score] + aux)
    score, aux = refreshed[0], refreshed[1:]

    rotated_score = ctx.engine.rotate(score, ctx.rotation_key, -half)
    diff = ctx.engine.subtract(rotated_score, score)
    dec_diff = ctx.engine.decrypt(diff, ctx.sk)[:half]
    normalized = ctx.engine.multiply(diff, 1.0 / normalizer)
    dec_norm = ctx.engine.decrypt(normalized, ctx.sk)[:half]

    sign = sharpened_sign(ctx, normalized, NODE5_SHARPEN)
    dec_sign = ctx.engine.decrypt(sign, ctx.sk)[:half]
    choose_first = ctx.engine.multiply(ctx.engine.add(sign, 1.0), 0.5)
    dec_choose = ctx.engine.decrypt(choose_first, ctx.sk)[:half]

    score = encrypted_blend(ctx, choose_first, score, rotated_score)
    new_aux = []
    for a in aux:
        rotated_a = ctx.engine.rotate(a, ctx.rotation_key, -half)
        new_aux.append(encrypted_blend(ctx, choose_first, a, rotated_a))
    aux = new_aux

    n_cur = half
    round_idx += 1
    dec_score = ctx.engine.decrypt(score, ctx.sk)[:n_cur]
    print(f"\n--- round{round_idx} (n={n_cur}) ---")
    print(f"  diff(raw)          ={[round(v.real,3) for v in dec_diff]}")
    print(f"  normalized(diff/norm)={[round(v.real,6) for v in dec_norm]}")
    print(f"  sign(sharpen={NODE5_SHARPEN}x)  ={[round(v.real,4) for v in dec_sign]}")
    print(f"  choose_first        ={[round(v.real,4) for v in dec_choose]}")
    print(f"  new score           ={[round(v.real,2) for v in dec_score]}")

feat_conf = [float(ctx.engine.decrypt(fc, ctx.sk)[0].real) for fc in aux[1 : 1 + n_features]]
thr = float(ctx.engine.decrypt(aux[0], ctx.sk)[0].real)
print(f"\n최종: threshold={thr:.4f} feature_confidence={[round(v,4) for v in feat_conf]}")
print(f"(참고: plaintext 기준 정답은 feature=3, threshold=0.5)")
