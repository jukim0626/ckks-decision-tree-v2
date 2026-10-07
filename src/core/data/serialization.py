"""프로세스 경계를 넘어 key/dataset ciphertext를 주고받기 위한 직렬화 유틸.

부모 프로세스(오케스트레이터)가 write_*로 한 번 저장해두면, 자식 프로세스(epoch/setup/
finalize worker)가 read_*로 똑같은 key/dataset을 복원해서 이어서 계산한다 - desilofhe가
key가 살아있는 동안 ciphertext 메모리 풀을 계속 붙잡고 있어서(개별 ciphertext를
del+gc.collect()해도 GPU 메모리가 거의 안 줄고, key까지 지워야 줄어듦) 노드/epoch별로
프로세스를 분리해야 GPU 메모리가 실제로 회수된다.

2026-09-09 리팩터로 closed_form_mgi/io_utils.py에서 분리됨."""

from __future__ import annotations

import sys
from pathlib import Path

from core.ckks_engine import BootstrapTrainingContext
from core.data.dataset import EncryptedDataset


def write_keys(ctx, keys_dir: Path, sk_path: Path | None = None) -> None:
    """keys_dir에는 서버가 연산에 쓰는 공개/평가용 key만 둔다. sk는 `sk_path`(보통
    session_dir/client/sk.bin)에 따로 저장한다.

    2026-10-07: 예전엔 sk도 keys_dir에 같이 저장하고 모든 worker가 읽었다 - 학습 worker는
    sk를 실제로 쓰지 않았지만, 서버 프로세스가 sk를 들고 있으면 "서버는 복호화할 수 없다"는
    위협 모델을 코드로 보장할 수 없어서 분리했다. sk_path를 안 주면 예전처럼 keys_dir에
    저장한다(archive 계보 호환용)."""
    keys_dir.mkdir(parents=True, exist_ok=True)
    if sk_path is None:
        sk_path = keys_dir / "sk.bin"
    sk_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.engine.write_secret_key(ctx.sk, sk_path)
    ctx.engine.write_public_key(ctx.pk, keys_dir / "pk.bin")
    ctx.engine.write_relinearization_key(ctx.rlk, keys_dir / "rlk.bin")
    ctx.engine.write_rotation_key(ctx.rotation_key, keys_dir / "rotk.bin")
    ctx.engine.write_conjugation_key(ctx.conjugation_key, keys_dir / "conjk.bin")
    ctx.engine.write_small_bootstrap_key(ctx.small_bootstrap_key, keys_dir / "sbk.bin")


def client_sk_path(session_dir: Path) -> Path:
    """client만 접근하는 secret key 위치."""
    return session_dir / "client" / "sk.bin"


def load_server_context(engine, keys_dir: Path, mode: str, device_id: int) -> BootstrapTrainingContext:
    """서버 측 worker(학습 epoch, encrypted inference forward)용 context - **sk를 읽지 않는다**
    (`ctx.sk is None`). 서버 코드 경로에 실수로 decrypt가 들어가면 바로 에러가 난다."""
    return BootstrapTrainingContext(
        engine=engine,
        sk=None,
        pk=engine.read_public_key(keys_dir / "pk.bin"),
        rlk=engine.read_relinearization_key(keys_dir / "rlk.bin"),
        rotation_key=engine.read_rotation_key(keys_dir / "rotk.bin"),
        conjugation_key=engine.read_conjugation_key(keys_dir / "conjk.bin"),
        small_bootstrap_key=engine.read_small_bootstrap_key(keys_dir / "sbk.bin"),
        mode=mode,
        device_id=device_id,
    )


def load_client_secret_key(engine, session_dir: Path):
    """client 측 코드(최종 score/파라미터 decrypt)만 호출한다. 2026-10-07 이전 세션은 sk가
    keys/sk.bin에 있으므로 그쪽으로 fallback한다."""
    path = client_sk_path(session_dir)
    if not path.exists():
        legacy = session_dir / "keys" / "sk.bin"
        if not legacy.exists():
            raise FileNotFoundError(f"secret key가 없습니다: {path} (구버전 위치 {legacy}도 없음)")
        print(f"[경고] 구버전 세션 - sk를 {legacy}에서 읽습니다.", file=sys.stderr)
        path = legacy
    return engine.read_secret_key(path)


def load_context(engine, keys_dir: Path, mode: str, device_id: int) -> BootstrapTrainingContext:
    """빈 Engine(키 생성 없이 만든)에 직렬화된 key를 읽어 채운 BootstrapTrainingContext를
    만든다. create_bootstrap_context()처럼 새로 키를 생성하지 않아 subprocess마다 불필요한
    keygen을 반복하지 않는다.

    **레거시(archive 계보 전용)**: keys_dir/sk.bin까지 읽는다. gradient_soft_tree는
    `load_server_context`(sk 없음) + `load_client_secret_key`를 쓴다."""
    return BootstrapTrainingContext(
        engine=engine,
        sk=engine.read_secret_key(keys_dir / "sk.bin"),
        pk=engine.read_public_key(keys_dir / "pk.bin"),
        rlk=engine.read_relinearization_key(keys_dir / "rlk.bin"),
        rotation_key=engine.read_rotation_key(keys_dir / "rotk.bin"),
        conjugation_key=engine.read_conjugation_key(keys_dir / "conjk.bin"),
        small_bootstrap_key=engine.read_small_bootstrap_key(keys_dir / "sbk.bin"),
        mode=mode,
        device_id=device_id,
    )


def write_dataset(ctx, dataset, dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for j, ct in enumerate(dataset.enc_features):
        ctx.engine.write_ciphertext(ct, dataset_dir / f"feature_{j}.ct")
    for c, ct in enumerate(dataset.enc_labels):
        ctx.engine.write_ciphertext(ct, dataset_dir / f"label_{c}.ct")


def read_dataset(ctx, dataset_dir: Path, n_features: int, n_classes: int, n_samples: int) -> EncryptedDataset:
    enc_features = [ctx.engine.read_ciphertext(dataset_dir / f"feature_{j}.ct") for j in range(n_features)]
    enc_labels = [ctx.engine.read_ciphertext(dataset_dir / f"label_{c}.ct") for c in range(n_classes)]
    return EncryptedDataset(
        enc_features=enc_features,
        enc_labels=enc_labels,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
    )
