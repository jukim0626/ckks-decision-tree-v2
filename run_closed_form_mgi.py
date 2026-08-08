"""closed_form_mgi.py(grid 없는 closed-form threshold + soft-MGI blend)를 실제 CKKS로
depth별로 돌려서 정확도/시간/연산 횟수를 실측. hard tournament, 기존 static-grid
soft-MGI(run_soft_mgi_stump.py)와 비교하기 위한 벤치마크.

score_normalizer는 고정(n_samples**2)만 씀 - encrypted min/max(soft-min)는 아직 미구현
(closed_form_mgi.py 상단 docstring 참고). depth>=2는 plaintext에서 확인된 대로 이
한계 때문에 정확도가 낮게 나올 것으로 예상 - 이것도 이번 실측의 목적 중 하나.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from client_assisted.dataset import load_scaled_dataset_subset, one_hot_encode
from closed_form_mgi import train_and_eval_closed_form_mgi_tree
from fully_encrypted_mgi_stump import create_bootstrap_context, encrypt_dataset

DATASET = sys.argv[1] if len(sys.argv) > 1 else "iris"
DEPTHS = [int(d) for d in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 3]
BETA = 30.0


class CountingEngineProxy:
    def __init__(self, engine, tracked_method_names):
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


def decrypt_scalar(ctx, ct) -> float:
    return float(ctx.engine.decrypt(ct, ctx.sk)[0].real)


def decrypt_vector(ctx, ct, n: int) -> np.ndarray:
    return np.array([v.real for v in ctx.engine.decrypt(ct, ctx.sk)[:n]])


def main():
    X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(DATASET, test_size=30)
    n_classes = len(class_names)
    y_train_oh = one_hot_encode(y_train, n_classes=n_classes)
    y_test_oh = one_hot_encode(y_test, n_classes=n_classes)
    print(
        f"[setup] dataset={DATASET} n_features={X_train.shape[1]} train={X_train.shape[0]} test={X_test.shape[0]}",
        flush=True,
    )

    for depth in DEPTHS:
        ctx = create_bootstrap_context(mode="gpu")
        counter = CountingEngineProxy(ctx.engine, ["sign_bootstrap", "bootstrap", "merge_bootstrap", "multiply"])
        ctx.engine = counter

        train_dataset = encrypt_dataset(ctx, X_train, y_train_oh)
        test_dataset = encrypt_dataset(ctx, X_test, y_test_oh)

        print(f"\n=== depth={depth} ===", flush=True)
        t0 = time.time()
        leaf_counts, leaf_test_weights = train_and_eval_closed_form_mgi_tree(
            ctx, train_dataset, test_dataset, depth, beta=BETA, verbose=True
        )
        elapsed = time.time() - t0

        n_test = X_test.shape[0]
        leaf_counts_dec = [np.array([decrypt_scalar(ctx, c) for c in lc]) for lc in leaf_counts]
        leaf_test_weights_dec = [decrypt_vector(ctx, w, n_test) for w in leaf_test_weights]

        scores = np.zeros((n_test, n_classes))
        for w, counts in zip(leaf_test_weights_dec, leaf_counts_dec):
            scores += w[:, None] * counts[None, :]
        preds = np.argmax(scores, axis=1)
        accuracy = float((preds == y_test).mean())

        print(f"[depth={depth}] time={elapsed:.2f}s accuracy={accuracy:.4f}")
        print(f"[depth={depth}] call counts={dict(counter.counts)}")
        print(f"[depth={depth}] leaf class distribution:")
        for i, c in enumerate(leaf_counts_dec):
            print(f"   leaf[{i}] counts={np.round(c,2)}")


if __name__ == "__main__":
    main()
