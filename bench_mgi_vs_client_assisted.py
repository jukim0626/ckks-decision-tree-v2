"""iris depth=1 stump 기준, 기존 client-assisted(decrypt 기반) 학습 vs
fully-encrypted MGI(sign_bootstrap argmin) 학습의 시간/정확도를 나란히 비교.
"""

from __future__ import annotations

import time

from client_assisted import (
    create_context,
    encrypt_dataset as ca_encrypt_dataset,
    load_scaled_dataset_subset,
    make_public_grid_candidates,
    make_small_public_threshold_grid,
    one_hot_encode,
    train_client_assisted_fixed_depth_tree,
)
from fully_encrypted_mgi_stump import (
    create_bootstrap_context,
    debug_decrypt_winner,
    encrypt_dataset as mgi_encrypt_dataset,
    plaintext_mgi_best_split,
    train_fully_encrypted_mgi_stump,
)

CANDIDATE_COUNT = 3  # 프로젝트 기본값 (CLAUDE.md)


def main() -> None:
    X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(
        dataset_name="iris", test_size=30
    )
    y_train_one_hot = one_hot_encode(y_train, n_classes=len(class_names))
    candidates = make_public_grid_candidates(
        n_features=X_train.shape[1],
        thresholds=make_small_public_threshold_grid(candidate_count=CANDIDATE_COUNT),
    )
    print(
        f"[setup] dataset=iris | candidates={len(candidates)} | "
        f"train_samples={X_train.shape[0]} | n_features={X_train.shape[1]}",
        flush=True,
    )

    best_idx, best_score = plaintext_mgi_best_split(X_train, y_train_one_hot, candidates)
    print(
        f"[plaintext MGI] best={candidates[best_idx]} score={best_score:.4f}",
        flush=True,
    )

    # --- 기존 client-assisted(decrypt 기반 weighted Gini) ---
    print("\n=== client-assisted (decrypt 기반) ===", flush=True)
    ctx_ca = create_context(mode="gpu", max_level=20)
    dataset_ca = ca_encrypt_dataset(ctx_ca, X_train, y_train_one_hot)
    t0 = time.time()
    model, selections = train_client_assisted_fixed_depth_tree(
        ctx_ca, dataset_ca, candidates, depth=1, verbose=False
    )
    ca_time = time.time() - t0
    sel = selections.selections[0]
    print(
        f"[client-assisted] time={ca_time:.2f}s | feature={sel.feature_idx} | "
        f"threshold={sel.threshold:.4f} | gini={sel.gini_score:.4f}",
        flush=True,
    )

    # --- fully-encrypted MGI (sign_bootstrap argmin) ---
    print("\n=== fully-encrypted MGI (client decrypt 없음) ===", flush=True)
    ctx_mgi = create_bootstrap_context(mode="gpu")
    dataset_mgi = mgi_encrypt_dataset(ctx_mgi, X_train, y_train_one_hot)
    t0 = time.time()
    winner = train_fully_encrypted_mgi_stump(ctx_mgi, dataset_mgi, candidates, sharpen_iterations=20)
    mgi_time = time.time() - t0
    decoded = debug_decrypt_winner(ctx_mgi, winner)
    print(
        f"[fully-encrypted] time={mgi_time:.2f}s | "
        f"threshold(decrypt)={decoded['threshold']:.4f} | score(decrypt)={decoded['score']:.4f} | "
        f"feature_selectors(decrypt)={[f'{v:.3f}' for v in decoded['feature_selectors']]}",
        flush=True,
    )

    print("\n=== 요약 ===", flush=True)
    print(f"plaintext MGI 정답: feature={candidates[best_idx].feature_idx} threshold={candidates[best_idx].threshold:.4f}")
    print(f"client-assisted 선택: feature={sel.feature_idx} threshold={sel.threshold:.4f} | {ca_time:.2f}s")
    picked_feature = max(range(len(decoded["feature_selectors"])), key=lambda i: decoded["feature_selectors"][i])
    print(f"fully-encrypted 선택: feature={picked_feature} threshold={decoded['threshold']:.4f} | {mgi_time:.2f}s")
    print(f"배수 차이: fully-encrypted가 client-assisted보다 {mgi_time / ca_time:.1f}배 느림")


if __name__ == "__main__":
    main()
