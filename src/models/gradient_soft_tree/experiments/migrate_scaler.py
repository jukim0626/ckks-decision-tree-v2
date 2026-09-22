"""scaler를 저장하지 않던 예전 코드로 만든 session_dir에, config.json에 남은 정보만으로
scaler를 **재구성**해서 session_dir/client/scaler.json으로 채워 넣는 명시적 migration
도구다. finalize_worker.py/predict_*.py가 scaler 파일이 없으면 조용히 다시 fit하는 대신
`FileNotFoundError`로 멈추도록 만든 것의 짝 - "복구하고 싶으면 이 스크립트를 명시적으로
실행하라"는 의도다.

**정확한 복원을 보장하지 않는다**: 이 스크립트는 config.json에 적힌 (dataset_name,
test_size, max_train)으로 `split_dataset_subset`을 다시 호출해서 그 시점과 "같아야 할"
train raw split을 얻은 뒤 새로 scaler를 fit한다. `random_state=42`가 고정이라 원본 데이터
fetch(sklearn 번들 데이터셋이나 OpenML fetch)와 sklearn 버전이 학습 당시와 동일하다면
원래 scaler와 동일하게 재현되지만, 그 전제가 깨지면(라이브러리 업데이트로 OpenML 응답이
바뀌거나 데이터 순서가 달라지는 등) 이 스크립트는 그 사실을 알 방법이 없다 - 그래서
"명시적 migration"이지 "자동 복구"가 아니다.

사용법: python -m models.gradient_soft_tree.experiments.migrate_scaler <session_dir>

2026-09-22: local_loss 계보가 제거되면서 이 도구가 다루는 건 baseline/packed가 쓰는
leaf_logits 파라미터 스키마 하나뿐이다(예전엔 local_logits 스키마도 구분해서 다뤘음)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data.dataset import fit_scaler, resolve_leaf_family_test_size, save_scaler, split_dataset_subset  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print("사용법: python -m models.gradient_soft_tree.experiments.migrate_scaler <session_dir>", file=sys.stderr)
        raise SystemExit(1)
    session_dir = Path(sys.argv[1])
    scaler_path = session_dir / "client" / "scaler.json"
    if scaler_path.exists():
        print(f"[migrate_scaler] 이미 scaler가 있습니다: {scaler_path} - 아무 것도 안 함.")
        return

    config = json.loads((session_dir / "config.json").read_text())
    dataset_name = config["dataset_name"]
    n_features = config["n_features"]
    max_train = config.get("max_train")

    test_size = resolve_leaf_family_test_size(config)
    print(
        f"[migrate_scaler] session={session_dir} dataset={dataset_name} "
        f"test_size={test_size} max_train={max_train}"
    )
    print(
        "[migrate_scaler] 경고: 이 scaler는 지금 이 환경(sklearn/OpenML fetch)으로 "
        "재구성한 것입니다 - 학습 당시와 데이터/라이브러리 버전이 다르면 원본과 다를 수 "
        "있고, 이 스크립트는 그 차이를 감지할 방법이 없습니다. 가능하면 세션을 다시 "
        "학습하는 쪽을 우선 고려하세요.",
        file=sys.stderr,
    )

    X_train_raw, _X_test_raw, _y_train, _y_test, _ = split_dataset_subset(
        dataset_name, test_size=test_size, max_train=max_train
    )
    if X_train_raw.shape[1] != n_features:
        raise ValueError(
            f"재구성한 train set의 feature 수({X_train_raw.shape[1]})가 config.json의 "
            f"n_features({n_features})와 다릅니다 - dataset loader가 학습 당시와 달라졌을 "
            "가능성이 있어 안전하게 재구성할 수 없습니다."
        )
    scaler = fit_scaler(X_train_raw)
    save_scaler(scaler, scaler_path, dataset_name=dataset_name)
    print(f"[migrate_scaler] 완료: {scaler_path}")


if __name__ == "__main__":
    main()
