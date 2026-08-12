"""프로세스 경계를 넘어 key/dataset ciphertext를 주고받기 위한 직렬화 유틸.

train.py(부모 프로세스)가 write_*로 한 번 저장해두면, node_worker.py/leaf_worker.py(자식
프로세스)가 read_*로 똑같은 key/dataset을 복원해서 이어서 계산한다 - desilofhe가 key가
살아있는 동안 ciphertext 메모리 풀을 계속 붙잡고 있어서(2026-08-11 실측: 개별 ciphertext를
del+gc.collect()해도 GPU 메모리가 거의 안 줄고, key까지 지워야 줄어듦) 노드별로 프로세스를
분리해야 GPU 메모리가 실제로 회수된다."""

from __future__ import annotations

from pathlib import Path

from client_assisted.dataset import EncryptedDataset


def write_keys(ctx, keys_dir: Path) -> None:
    keys_dir.mkdir(parents=True, exist_ok=True)
    ctx.engine.write_secret_key(ctx.sk, keys_dir / "sk.bin")
    ctx.engine.write_public_key(ctx.pk, keys_dir / "pk.bin")
    ctx.engine.write_relinearization_key(ctx.rlk, keys_dir / "rlk.bin")
    ctx.engine.write_rotation_key(ctx.rotation_key, keys_dir / "rotk.bin")
    ctx.engine.write_conjugation_key(ctx.conjugation_key, keys_dir / "conjk.bin")
    ctx.engine.write_small_bootstrap_key(ctx.small_bootstrap_key, keys_dir / "sbk.bin")


def load_context(engine, keys_dir: Path, mode: str, device_id: int):
    """빈 Engine(키 생성 없이 만든)에 직렬화된 key를 읽어 채운 BootstrapTrainingContext를
    만든다. create_bootstrap_context()처럼 새로 키를 생성하지 않아 subprocess마다 불필요한
    keygen을 반복하지 않는다."""
    from closed_form_mgi.primitives import BootstrapTrainingContext

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
