

from __future__ import annotations

from dataclasses import dataclass

from desilofhe import Engine


@dataclass
class EncryptedTrainingContext:


    engine: Engine
    sk: object
    pk: object
    rlk: object
    rotation_key: object
    mode: str
    device_id: int


def create_context(
    mode: str = "gpu",
    device_id: int = 0,
    max_level: int = 20,
    slot_count: int | None = None,
) -> EncryptedTrainingContext:
    """slot_count를 주면 그 값으로 ciphertext 크기를 고정한다 (논문은 poly modulus degree
    8192 -> slot_count=4096 packing을 씀; 안 주면 기존처럼 max_level 기준 자동 결정)."""
    if slot_count is None:
        engine = Engine(max_level=max_level, mode=mode, device_id=device_id, compact=True)
    else:
        engine = Engine(
            slot_count=slot_count, max_level=max_level, mode=mode, device_id=device_id, compact=True
        )
    sk = engine.create_secret_key()
    pk = engine.create_public_key(sk)
    rlk = engine.create_relinearization_key(sk)
    rotation_key = engine.create_rotation_key(sk)
    return EncryptedTrainingContext(
        engine=engine,
        sk=sk,
        pk=pk,
        rlk=rlk,
        rotation_key=rotation_key,
        mode=mode,
        device_id=device_id,
    )


def sync_engine_if_needed(ctx: EncryptedTrainingContext) -> None:

    try:
        ctx.engine.sync()
    except RuntimeError as exc:
        if "Async GPU mode" not in str(exc):
            raise
