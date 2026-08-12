"""closed_form_mgi.train/inference(grid 없는 closed-form threshold + soft-MGI blend)를
실제 CKKS로 depth별로 돌려서 정확도/시간/연산 횟수를 실측. hard tournament, 기존
static-grid soft-MGI(closed_form_mgi 이전 버전, archive 참고)와 비교하기 위한 벤치마크.

score_normalizer는 고정(n_samples**2)만 씀 - encrypted min/max(soft-min)는 아직 미구현
(closed_form_mgi/train.py 상단 docstring 참고). depth>=2는 plaintext에서 확인된 대로 이
한계 때문에 정확도가 낮게 나올 것으로 예상 - 이것도 이번 실측의 목적 중 하나.

실행: 프로젝트 루트에서 `python -m closed_form_mgi.run [dataset] [depths]`
(예: `python -m closed_form_mgi.run iris 1,2,3`)
"""

from __future__ import annotations

import shutil
import sys
import time

import numpy as np

from client_assisted.dataset import load_scaled_dataset_subset, one_hot_encode
from closed_form_mgi.inference import route_dataset_through_model, score_and_predict
from closed_form_mgi.primitives import create_bootstrap_context, encrypt_dataset
from closed_form_mgi.train import train_closed_form_mgi_tree

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
        model = train_closed_form_mgi_tree(ctx, train_dataset, depth, beta=BETA, verbose=True)
        leaf_test_weights = route_dataset_through_model(ctx, model, test_dataset)
        n_test = X_test.shape[0]
        preds = score_and_predict(ctx, model, leaf_test_weights, n_test, n_classes)
        elapsed = time.time() - t0

        accuracy = float((preds == y_test).mean())
        leaf_counts_dec = [
            np.array([decrypt_scalar(ctx, ctx.engine.read_ciphertext(c)) for c in lc]) for lc in model.leaf_counts
        ]

        print(f"[depth={depth}] time={elapsed:.2f}s accuracy={accuracy:.4f}")
        print(f"[depth={depth}] call counts={dict(counter.counts)}")
        print(f"[depth={depth}] leaf class distribution:")
        for i, c in enumerate(leaf_counts_dec):
            print(f"   leaf[{i}] counts={np.round(c, 2)}")

        shutil.rmtree(model.session_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
