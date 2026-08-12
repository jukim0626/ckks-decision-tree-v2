"""노드 하나의 학습 연산(threshold + soft-MGI weight + blended gate)을 독립 프로세스로
실행한다. train_closed_form_mgi_tree()가 노드마다 `python -m closed_form_mgi.node_worker
<session_dir> <node_id>`로 이 스크립트를 실행하고, 끝나면 프로세스가 죽으면서 OS가 GPU
메모리를 강제로 회수한다 (2026-08-11: desilofhe가 key가 살아있는 동안 ciphertext 메모리
풀을 계속 붙잡고 있어서, 같은 프로세스 안에서 트리 전체를 학습하면 del+gc.collect()로는 GPU
메모리가 안 풀리고 depth>=2에서 OOM이 났던 문제를 프로세스 경계로 우회한 것 - 상세 배경은
train.py의 train_closed_form_mgi_tree docstring 참고).

session_dir/nodes/<node_id>/input_weights.ct를 읽어서, 같은 디렉터리에
threshold_{j}.ct(feature별), weights.ct, left_weights.ct, right_weights.ct를 쓴다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from closed_form_mgi.io_utils import load_context, read_dataset
from closed_form_mgi.primitives import create_bootstrap_engine
from closed_form_mgi.train import process_single_node


def main() -> None:
    session_dir = Path(sys.argv[1])
    node_id = sys.argv[2]
    config = json.loads((session_dir / "config.json").read_text())

    engine = create_bootstrap_engine(mode=config["mode"], device_id=config["device_id"])
    ctx = load_context(engine, session_dir / "keys", mode=config["mode"], device_id=config["device_id"])
    dataset = read_dataset(
        ctx,
        session_dir / "dataset",
        n_features=config["n_features"],
        n_classes=config["n_classes"],
        n_samples=config["n_samples"],
    )

    node_dir = session_dir / "nodes" / node_id
    enc_train_weights = ctx.engine.read_ciphertext(node_dir / "input_weights.ct")

    thresholds, weights, left_train, right_train = process_single_node(
        ctx, dataset, enc_train_weights, beta=config["beta"], score_normalizer=config["score_normalizer"]
    )

    for j, threshold in enumerate(thresholds):
        ctx.engine.write_ciphertext(threshold, node_dir / f"threshold_{j}.ct")
    ctx.engine.write_ciphertext(weights, node_dir / "weights.ct")
    ctx.engine.write_ciphertext(left_train, node_dir / "left_weights.ct")
    ctx.engine.write_ciphertext(right_train, node_dir / "right_weights.ct")


if __name__ == "__main__":
    main()
