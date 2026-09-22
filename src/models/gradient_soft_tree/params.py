"""alpha/threshold/leaf_logits(baseline·packed) ciphertext를 session_dir/params/ 아래
파일로 읽고 쓰는 공통 I/O.

파일명 규칙(alpha_{i}.ct, threshold_{i}_{j}.ct, leaf_{l}.ct)과 직렬화 포맷
(engine.read_ciphertext/write_ciphertext)은 전부 기존 그대로 유지한다 - setup/epoch/
finalize worker가 프로세스 경계를 넘어 이 파일들로 파라미터를 주고받는 계약 자체
(=파일명, 어떤 worker가 언제 읽고 쓰는지)는 이 리팩터로 바뀌지 않는다. packed가
baseline의 setup_worker/finalize_worker를 그대로 재사용할 수 있는 것도 이 계약이
그대로라서다.

2026-09-21 1차 리팩터: 이 read/write 루프가 baseline/{setup,epoch,finalize}_worker.py,
packed/epoch_worker_packed.py에 거의 동일하게 반복돼 있던 걸 통합했다(수치 동작 변경
없음 - 어떤 ciphertext도 다르게 읽거나 쓰지 않고, 파일 I/O 순서만 alpha 전부/threshold
전부로 묶였을 뿐 각 파일은 독립적인 쓰기라 순서가 결과에 영향을 주지 않는다).

2026-09-22: local_logits 계열(local_{level}_{k}.ct, load/save_local_logits)은 local_loss
계보 제거와 함께 삭제 - 더 이상 어떤 worker도 이 파일명을 쓰지 않는다. opt는
leaf_logits 계열(baseline과 동일 스키마)을 썼으나 opt 계보 자체가 제거됐다."""

from __future__ import annotations

from pathlib import Path


def load_alpha(engine, params_dir: Path, n_internal: int) -> list:
    return [engine.read_ciphertext(params_dir / f"alpha_{i}.ct") for i in range(n_internal)]


def save_alpha(engine, params_dir: Path, alpha_cts: list) -> None:
    for i, ct in enumerate(alpha_cts):
        engine.write_ciphertext(ct, params_dir / f"alpha_{i}.ct")


def load_threshold(engine, params_dir: Path, n_internal: int, n_features: int) -> list:
    return [
        [engine.read_ciphertext(params_dir / f"threshold_{i}_{j}.ct") for j in range(n_features)]
        for i in range(n_internal)
    ]


def save_threshold(engine, params_dir: Path, threshold_cts: list) -> None:
    for i, row in enumerate(threshold_cts):
        for j, ct in enumerate(row):
            engine.write_ciphertext(ct, params_dir / f"threshold_{i}_{j}.ct")


def load_leaf_logits(engine, params_dir: Path, n_leaves: int) -> list:
    return [engine.read_ciphertext(params_dir / f"leaf_{l}.ct") for l in range(n_leaves)]


def save_leaf_logits(engine, params_dir: Path, leaf_cts: list) -> None:
    for l, ct in enumerate(leaf_cts):
        engine.write_ciphertext(ct, params_dir / f"leaf_{l}.ct")


