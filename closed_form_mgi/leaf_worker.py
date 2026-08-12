"""학습이 끝난 뒤 leaf들의 최종 weighted class count를 별도 프로세스에서 계산한다.
node_worker.py만큼 반복이 무겁진 않지만(뉴턴-랩슨 루프 없이 multiply+sum뿐), 일관성을
위해 마찬가지로 프로세스를 분리한다 (배경은 train.py의 train_closed_form_mgi_tree
docstring 참고).

session_dir/nodes/<leaf_id>/input_weights.ct(그 leaf에 도달한 최종 encrypted weight)를
읽어서, 같은 디렉터리에 count_{c}.ct(class별)를 쓴다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from client_assisted.server_ops import encrypted_class_counts
from closed_form_mgi.io_utils import load_context, read_dataset
from closed_form_mgi.primitives import create_bootstrap_engine


def main() -> None:
    session_dir = Path(sys.argv[1])
    leaf_ids = sys.argv[2].split(",")
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

    for leaf_id in leaf_ids:
        leaf_dir = session_dir / "nodes" / leaf_id
        leaf_weights = ctx.engine.read_ciphertext(leaf_dir / "input_weights.ct")
        counts = encrypted_class_counts(ctx, leaf_weights, dataset.enc_labels)
        for c, count in enumerate(counts):
            ctx.engine.write_ciphertext(count, leaf_dir / f"count_{c}.ct")


if __name__ == "__main__":
    main()
