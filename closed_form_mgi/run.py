"""closed_form_mgi.train/inference(grid 없는 closed-form threshold + soft-MGI blend)를
실제 CKKS로 depth별로 돌려서 정확도/시간을 실측. hard tournament, 기존 static-grid
soft-MGI(closed_form_mgi 이전 버전, archive 참고)와 비교하기 위한 벤치마크.

score_normalizer는 고정(n_samples**2)만 씀 - encrypted min/max(soft-min)는 아직 미구현
(closed_form_mgi/train.py 상단 docstring 참고). depth>=2는 plaintext에서 확인된 대로 이
한계 때문에 정확도가 낮게 나올 것으로 예상 - 이것도 이번 실측의 목적 중 하나.

실행: 프로젝트 루트에서 `python -m closed_form_mgi.run [dataset] [depths]`
(예: `python -m closed_form_mgi.run iris 1,2,3`)

**depth마다 별도 프로세스**(depth_worker.py)를 띄운다 - desilofhe의 GPU 메모리 풀은 한
프로세스 안에서 한 번 늘어나면 `del`을 해도 그 프로세스가 살아있는 동안 절대 안 줄어드는
구조라, depth 1,2,3을 한 프로세스에서 순차 처리하면 뒤로 갈수록 baseline이 계속 높아지다가
결국 OOM 난다 (원인 규명 과정은 EXPERIMENT_LOG.md 2026-08-11/12, depth_worker.py docstring
참고). node_worker.py가 노드 단위로 하는 격리를 depth 단위로 확장한 것.
"""

from __future__ import annotations

import subprocess
import sys

from client_assisted.dataset import load_scaled_dataset_subset

DATASET = sys.argv[1] if len(sys.argv) > 1 else "iris"
DEPTHS = [int(d) for d in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 3]


def main() -> None:
    X_train, X_test, y_train, y_test, class_names = load_scaled_dataset_subset(DATASET, test_size=30)
    print(
        f"[setup] dataset={DATASET} n_features={X_train.shape[1]} train={X_train.shape[0]} test={X_test.shape[0]}",
        flush=True,
    )

    for depth in DEPTHS:
        result = subprocess.run([sys.executable, "-m", "closed_form_mgi.depth_worker", DATASET, str(depth)])
        if result.returncode != 0:
            raise RuntimeError(f"depth={depth} 실패 (closed_form_mgi.depth_worker exit code {result.returncode})")


if __name__ == "__main__":
    main()
