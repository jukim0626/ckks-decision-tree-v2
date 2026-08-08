"""soft-MGI plaintext(soft_mgi_plaintext.py) 성능 sweep: dataset x depth x beta.

CKKS로 포팅하기 전에 plaintext에서 빠르게 좋은 설정(특히 beta, depth)을 찾기 위한 실험.
비교 대상 3개:
  - soft-MGI: 후보 K개를 softmax weight로 blend (진짜 실험 대상)
  - hard-MGI-argmin: 같은 candidate pool/재귀 구조, argmin으로 1개만 고름 (soft-MGI와
    candidate 풀/재귀 구조가 완전히 동일해서 순수하게 "섞을지 말지"의 효과만 비교 가능)
  - sklearn DecisionTree(gini): 외부 참조용 상한선 (soft/hard 둘 다 sigmoid 근사+public
    grid라는 제약이 있어서 sklearn보다 낮은 게 정상, 격차 크기를 가늠하는 용도)
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from client_assisted.candidates import make_public_grid_candidates, make_small_public_threshold_grid
from client_assisted.dataset import load_scaled_dataset_subset, one_hot_encode
from soft_mgi_plaintext import predict, train_hard_mgi_argmin, train_soft_mgi

DATASETS = ["iris", "wine", "breast_cancer"]
DEPTHS = [1, 2, 3]
# beta<=50: 실제 CKKS exp(-beta*x) 다항식 근사(exp_approx_coeffs.py, degree=20)가 지금
# 검증된 범위. 그 이상은 cheb2poly가 수치적으로 붕괴해서 이 sweep에서도 CKKS로 포팅 가능한
# 범위로만 제한한다.
BETAS = [1.0, 4.0, 10.0, 20.0, 30.0, 50.0]
NORMALIZER_MODES = ["fixed", "range"]
CANDIDATE_COUNT = 3

results = []

for dataset_name in DATASETS:
    X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(
        dataset_name=dataset_name, test_size=30
    )
    n_classes = len(class_names)
    y_train_oh = one_hot_encode(y_train, n_classes=n_classes)
    candidates = make_public_grid_candidates(
        n_features=X_train.shape[1],
        thresholds=make_small_public_threshold_grid(candidate_count=CANDIDATE_COUNT),
    )

    for depth in DEPTHS:
        sk_tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=0)
        sk_tree.fit(X_train, y_train)
        sk_acc = float((sk_tree.predict(X_test) == y_test).mean())

        hard_root = train_hard_mgi_argmin(X_train, y_train_oh, candidates, depth)
        hard_pred = predict(X_test, hard_root, candidates, n_classes)
        hard_acc = float((hard_pred == y_test).mean())

        row = {
            "dataset": dataset_name,
            "depth": depth,
            "sklearn_gini_acc": sk_acc,
            "hard_mgi_argmin_acc": hard_acc,
        }

        for mode in NORMALIZER_MODES:
            for beta in BETAS:
                soft_root = train_soft_mgi(
                    X_train, y_train_oh, candidates, depth, beta, normalizer_mode=mode
                )
                soft_pred = predict(X_test, soft_root, candidates, n_classes)
                soft_acc = float((soft_pred == y_test).mean())
                row[f"soft_{mode}_beta{beta:g}_acc"] = soft_acc

        results.append(row)
        for mode in NORMALIZER_MODES:
            beta_accs = " ".join(
                f"b{b:g}={row[f'soft_{mode}_beta{b:g}_acc']:.3f}" for b in BETAS
            )
            print(
                f"[{dataset_name:14s} depth={depth} {mode:5s}] "
                f"sklearn={sk_acc:.3f} hard_argmin={hard_acc:.3f} | {beta_accs}",
                flush=True,
            )

with open("soft_mgi_plaintext_sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nsaved -> soft_mgi_plaintext_sweep_results.json")
