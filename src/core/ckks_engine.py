"""CKKS bootstrap 엔진/컨텍스트 생성 + level 관리. 모든 모델 계보(gradient_soft_tree의
baseline/opt/packed/local_loss, 그리고 archive의 closed_form_mgi)가 공유하는 프리미티브.

2026-09-09 리팩터로 closed_form_mgi/primitives.py에서 분리됨(원래 이 파일 안에
MGI 전용 로직과 뒤섞여 있었음) - 원본 이력/주석은 git log로 확인 가능."""

from __future__ import annotations

from dataclasses import dataclass

from desilofhe import Engine


@dataclass
class BootstrapTrainingContext:
    """bootstrap을 쓰는 모든 학습 실험이 공유하는 context."""

    engine: Engine
    sk: object
    pk: object
    rlk: object
    rotation_key: object
    conjugation_key: object
    small_bootstrap_key: object
    mode: str
    device_id: int
    # 2026-10-09: 일반 bootstrap key(desilofhe create_bootstrap_key, medium ~9.9GiB). 있으면 ensure_level이
    # small key 대신 이걸 쓴다 - GPU 실측 bootstrap 4.4s -> 1.5s(merge_bootstrap이면 ct 2개에 1.5s).
    bootstrap_key: object = None


def create_bootstrap_engine(mode: str = "gpu", device_id: int = 0, level_preset: int | None = None) -> Engine:
    """level_preset: None(기본 26-level, closed_form_mgi가 정확도 유지에 필요했던 값),
    17 또는 14(gradient_soft_tree처럼 더 순한 조건에서 쓸 수 있는 더 가벼운 프리셋).
    use_bootstrap과 use_bootstrap_to_{14,17}_levels는 상호 배타적 플래그라 level_preset이
    있으면 use_bootstrap 자체를 빼야 한다."""
    if level_preset is None:
        return Engine(mode=mode, use_bootstrap=True, device_id=device_id)
    if level_preset == 17:
        return Engine(mode=mode, use_bootstrap_to_17_levels=True, device_id=device_id)
    if level_preset == 14:
        return Engine(mode=mode, use_bootstrap_to_14_levels=True, device_id=device_id)
    raise ValueError(f"unsupported level_preset: {level_preset!r} (expected None, 17, or 14)")


def create_bootstrap_context(
    mode: str = "gpu", device_id: int = 0, level_preset: int | None = None, regular_bootstrap_key: bool = False
) -> BootstrapTrainingContext:
    engine = create_bootstrap_engine(mode=mode, device_id=device_id, level_preset=level_preset)
    sk = engine.create_secret_key()
    pk = engine.create_public_key(sk)
    rlk = engine.create_relinearization_key(sk)
    rotation_key = engine.create_rotation_key(sk)
    conjugation_key = engine.create_conjugation_key(sk)
    small_bootstrap_key = engine.create_small_bootstrap_key(sk)
    return BootstrapTrainingContext(
        engine=engine,
        sk=sk,
        pk=pk,
        rlk=rlk,
        rotation_key=rotation_key,
        conjugation_key=conjugation_key,
        small_bootstrap_key=small_bootstrap_key,
        mode=mode,
        device_id=device_id,
        bootstrap_key=engine.create_bootstrap_key(sk, size="medium") if regular_bootstrap_key else None,
    )


def _bootstrap(ctx, ct):
    if getattr(ctx, "bootstrap_key", None) is not None:
        return ctx.engine.bootstrap(ct, ctx.rlk, ctx.conjugation_key, ctx.bootstrap_key)
    return ctx.engine.bootstrap(ct, ctx.rlk, ctx.conjugation_key, ctx.rotation_key, ctx.small_bootstrap_key)


def _merge_bootstrap(ctx, a, b):
    if getattr(ctx, "bootstrap_key", None) is not None:
        return ctx.engine.merge_bootstrap(a, b, ctx.rlk, ctx.conjugation_key, ctx.bootstrap_key)
    return ctx.engine.merge_bootstrap(a, b, ctx.rlk, ctx.conjugation_key, ctx.rotation_key, ctx.small_bootstrap_key)


def ensure_levels(ctx, cts: list, min_level: int = 8) -> list:
    """여러 ciphertext를 한꺼번에 레벨 확보. bootstrap이 필요한 것들을 2개씩 merge_bootstrap으로 묶는다
    (실수 값 두 개를 실수부/허수부로 합쳐 bootstrap 한 번 - GPU 실측 1회 비용으로 ct 2개). 홀수면 마지막은 단독.
    엔진이 merge_bootstrap을 지원하지 않으면(numpy 시뮬레이터 등) 하나씩 한다."""
    out = list(cts)
    need = [i for i, ct in enumerate(out) if ct.level < min_level]
    merge = hasattr(ctx.engine, "merge_bootstrap")
    while need:
        if merge and len(need) >= 2:
            i, j = need.pop(0), need.pop(0)
            a, b = _merge_bootstrap(ctx, ctx.engine.intt(out[i]), ctx.engine.intt(out[j]))
            out[i], out[j] = ctx.engine.intt(a), ctx.engine.intt(b)
        else:
            i = need.pop(0)
            out[i] = ctx.engine.intt(_bootstrap(ctx, ctx.engine.intt(out[i])))
    return out


def ensure_level(ctx, ct, min_level: int = 8):
    """level이 min_level 밑으로 떨어진 ciphertext를 bootstrap으로 복구.

    bootstrap()은 입력이 NTT form이면 거부한다("should not be in NTT form") - ct-ct
    multiply 체인을 여러 단계 거친 ciphertext는 NTT form일 수 있어서, 호출 전에 무조건
    intt로 정규화한다(intt는 이미 normal form이어도 값이 안 깨지는 안전한 연산)."""
    if ct.level < min_level:
        return ctx.engine.intt(_bootstrap(ctx, ctx.engine.intt(ct)))
    return ct
