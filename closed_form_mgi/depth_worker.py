"""dataset+depth 하나를 통째로(학습+추론+정확도 출력) 별도 프로세스에서 처리한다.
`run.py`가 depth마다 이 스크립트를 subprocess로 띄운다.

**배경(EXPERIMENT_LOG.md 2026-08-11/12)**: 노드별로 프로세스를 분리(node_worker.py)해도
depth=2가 여전히 OOM 났다. 원인을 depth=1을 먼저 완주(학습+추론)한 뒤 depth=2로 넘어가는
실험으로 재현: depth=1 학습 직후 부모 GPU 메모리는 baseline(~4546MiB)이었지만, depth=1의
**추론**(route_dataset_through_model - 이건 subprocess 격리가 안 돼 있어 부모에서 직접
GPU 연산을 함)이 끝나면 5442MiB로 오르고, 그 뒤 ctx/model 등을 전부 `del`해도 5246MiB까지만
내려가고 원래 baseline으로 안 돌아온다. 이 상태에서 depth=2가 새로 keygen을 하면 **그
5246 위에 또 쌓여서** 9764MiB에서 시작한다. 즉 desilofhe의 GPU 메모리 풀은 한 프로세스
안에서 한 번 늘어나면 `del`을 해도 그 프로세스가 살아있는 동안 절대 안 줄어드는 구조라
(흔한 CUDA caching allocator 설계), depth를 여러 개 한 프로세스에서 순차 처리하면 뒤로
갈수록 시작 baseline이 계속 높아지다가 결국 OOM 난다. depth 하나당 프로세스를 통째로
분리해서, 매 depth가 완전히 깨끗한(baseline) GPU 상태에서 시작하게 만드는 게 이 파일의
목적이다 (node_worker.py가 노드 단위로 하는 것과 정확히 같은 원리를 depth 단위로 확장).

실행: python -m closed_form_mgi.depth_worker <dataset> <depth>
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

BETA = 30.0


def decrypt_scalar(ctx, ct) -> float:
    return float(ctx.engine.decrypt(ct, ctx.sk)[0].real)


def main() -> None:
    dataset_name = sys.argv[1]
    depth = int(sys.argv[2])

    X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(dataset_name, test_size=30)
    n_classes = len(class_names)
    y_train_oh = one_hot_encode(y_train, n_classes=n_classes)
    y_test_oh = one_hot_encode(y_test, n_classes=n_classes)

    ctx = create_bootstrap_context(mode="gpu")
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
    print(f"[depth={depth}] leaf class distribution:")
    for i, c in enumerate(leaf_counts_dec):
        print(f"   leaf[{i}] counts={np.round(c, 2)}")

    shutil.rmtree(model.session_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
