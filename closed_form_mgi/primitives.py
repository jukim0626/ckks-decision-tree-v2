"""Fully-encrypted MGI stump: client decrypt 없이 server가 MGI score 계산 + sign_bootstrap
기반 encrypted argmin까지 끝내는 depth=1 학습 실험.

docs/2024-529.pdf(HBDT, ESORICS 2024)의 division-free MGI criterion
(M G(S|A) ∝ sum_t(|S_t|^2 - sum_c|S_t,c|^2), Eq.3)과, desilofhe에 내장된
sign_bootstrap()(논문의 ApproxSign에 해당하는 CKKS comparison primitive)을 써서,
archive/encrypted_mgii_stump.py가 예전에 막혔던 지점(저정밀 sigmoid 기반 soft argmin
selector, EXPERIMENT_LOG.md "Tournament Argmin Progress" 참고)을 다시 시도한다.

paper의 FindMaxGroupPos(O(log n) SIMD grouping)까지는 아직 구현하지 않고, 우선 O(n)
순차 pairwise fold argmin으로 "client decrypt 없이도 sharp한 선택이 되는지"부터
검증한다. client_assisted 패키지가 쓰는 max_level 기반 Engine과 달리 sign_bootstrap은
Engine(mode=..., use_bootstrap=True)로 만든 별도 engine이 필요해서 context를 새로
정의한다 (필드명은 EncryptedTrainingContext와 맞춰서 encrypt_dataset() 등 기존 함수를
그대로 재사용).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from desilofhe import Engine

from ckks_tree import sigmoid_approx_enc
from client_assisted.candidates import (
    PublicSplitCandidate,
    make_public_grid_candidates,
    make_small_public_threshold_grid,
)
from client_assisted.dataset import (
    EncryptedDataset,
    encrypt_dataset,
    load_scaled_dataset_subset,
    one_hot_encode,
)


@dataclass
class BootstrapTrainingContext:
    """sign_bootstrap을 쓰는 fully-encrypted 실험 전용 context."""

    engine: Engine
    sk: object
    pk: object
    rlk: object
    rotation_key: object
    conjugation_key: object
    small_bootstrap_key: object
    mode: str
    device_id: int


def create_bootstrap_engine(mode: str = "gpu", device_id: int = 0) -> Engine:
    # 2026-08-11에 max_level을 26->17로 줄여봤다가(OOM 완화 목적) 되돌림: slot_count는
    # 실측상 메모리에 영향이 없고(bootstrap 파라미터셋이 slot_count와 무관하게 물리 ring
    # 크기가 고정됨), max_level은 26/17/14 세 프리셋뿐이라 중간값이 없는데, 17로 낮추면
    # production beta=30에서 soft_mgi_weights의 뉴턴-랩슨이 (레벨 여유가 빡빡해져 bootstrap이
    # 훨씬 자주 걸리면서) "ciphertext 값 크기 2~5 넘으면 bootstrap이 조용히 깨진다"는 기존
    # 버그를 다시 건드려 완전히 발산했다(디버그: debug_closed_form_pipeline.py로 재현,
    # weights가 1e90 스케일로 붕괴). 정확도를 지키려면 26(기본)을 써야 한다 - OOM은 파라미터가
    # 아니라 노드 처리 도중 GPU 메모리가 계속 쌓이는 별도의 leak으로 봐야 함.
    return Engine(mode=mode, use_bootstrap=True, device_id=device_id)


def create_bootstrap_context(mode: str = "gpu", device_id: int = 0) -> BootstrapTrainingContext:
    engine = create_bootstrap_engine(mode=mode, device_id=device_id)
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
    )


@dataclass
class TournamentWinner:
    """encrypted argmin의 현재 winner 상태. candidate index는 절대 복호화하지 않는다."""

    score: object
    threshold: object
    feature_selectors: list
    left_counts: list
    right_counts: list


def encrypted_weighted_class_counts(ctx, enc_weights, enc_labels: list) -> list:
    counts = []
    for enc_label in enc_labels:
        weighted = ctx.engine.multiply(enc_weights, enc_label, ctx.rlk)
        counts.append(ctx.engine.sum(weighted, ctx.rotation_key))
    return counts


def encrypted_mgi_from_counts(ctx, class_counts: list) -> object:
    """|D|^2 - sum_c |D_c|^2 (division-free MGI, docs/2024-529.pdf Eq.3)."""
    total = class_counts[0]
    for count in class_counts[1:]:
        total = ctx.engine.add(total, count)
    squared_total = ctx.engine.multiply(total, total, ctx.rlk)
    squared_sum = ctx.engine.multiply(class_counts[0], class_counts[0], ctx.rlk)
    for count in class_counts[1:]:
        squared_sum = ctx.engine.add(squared_sum, ctx.engine.multiply(count, count, ctx.rlk))
    return ctx.engine.subtract(squared_total, squared_sum)


def evaluate_candidate(ctx, dataset: EncryptedDataset, candidate: PublicSplitCandidate):
    """candidate 하나의 encrypted MGI score(left+right)와 left/right class counts."""
    enc_feature = dataset.enc_features[candidate.feature_idx]
    enc_diff = ctx.engine.subtract(enc_feature, candidate.threshold)
    right_prob = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_diff)
    left_prob = ctx.engine.subtract(1.0, right_prob)
    left_counts = encrypted_weighted_class_counts(ctx, left_prob, dataset.enc_labels)
    right_counts = encrypted_weighted_class_counts(ctx, right_prob, dataset.enc_labels)
    score = ctx.engine.add(
        encrypted_mgi_from_counts(ctx, left_counts),
        encrypted_mgi_from_counts(ctx, right_counts),
    )
    return score, left_counts, right_counts


def candidate_winner_state(ctx, candidate, score, left_counts, right_counts, n_features) -> TournamentWinner:
    feature_selectors = [
        ctx.engine.encrypt([1.0 if i == candidate.feature_idx else 0.0], ctx.pk)
        for i in range(n_features)
    ]
    threshold = ctx.engine.encrypt([candidate.threshold], ctx.pk)
    return TournamentWinner(
        score=score,
        threshold=threshold,
        feature_selectors=feature_selectors,
        left_counts=left_counts,
        right_counts=right_counts,
    )


def ensure_level(ctx, ct, min_level: int = 8):
    """level이 min_level 밑으로 떨어진 ciphertext를 bootstrap으로 복구.

    tournament fold를 반복할 때 score/threshold/counts는 sign_bootstrap과 달리 매 라운드
    encrypted_blend의 multiply 2번(level -2)만 소모하고 한 번도 복구가 안 되므로, 라운드가
    쌓이면 결국 더 이상 곱셈을 할 level이 안 남아서 sign_bootstrap 자체가 터진다
    ("the level of the input ciphertext is less than the target level"). candidate 수가
    많은 실제 데이터셋(iris candidates=12)에서 처음 겪은 문제라 넣었다.
    """
    if ct.level < min_level:
        # bootstrap()은 입력이 NTT form이면 거부한다("should not be in NTT form") - ct-ct
        # multiply 체인(encrypted_blend, server_compute_weighted_child_weights 등)을 여러
        # 단계 거친 ciphertext는 NTT form일 수 있어서, 호출 전에 무조건 intt로 정규화한다
        # (intt는 이미 normal form이어도 값이 안 깨지는 안전한 연산으로 확인함 - 여러 번
        # 걸어도 decrypt 결과 동일). depth=3 tree의 child node에서 실제로 이 문제로 막힘.
        ct = ctx.engine.intt(ct)
        refreshed = ctx.engine.bootstrap(
            ct, ctx.rlk, ctx.conjugation_key, ctx.rotation_key, ctx.small_bootstrap_key
        )
        return ctx.engine.intt(refreshed)
    return ct


def encrypted_blend(ctx, choose_left, left_value, right_value):
    """choose_left가 1에 가까우면 left_value, 0에 가까우면 right_value."""
    left_value = ensure_level(ctx, left_value)
    right_value = ensure_level(ctx, right_value)
    left_piece = ctx.engine.multiply(choose_left, left_value, ctx.rlk)
    right_weight = ctx.engine.subtract(1.0, choose_left)
    right_piece = ctx.engine.multiply(right_weight, right_value, ctx.rlk)
    return ctx.engine.add(left_piece, right_piece)


def sharpened_sign(ctx, enc_x, iterations: int):
    """sign_bootstrap(x)를 iterations번 반복 합성해서 0 근처 기울기(~pi/2>1)로 부호를
    +-1 쪽으로 밀어붙인다.

    sign_bootstrap 자체가 sin(pi/2 * x) 모양이라(=1 근방 밖에서는 주기적으로 aliasing되므로
    입력을 반드시 [-1,1] 안으로 정규화해야 함), 입력이 작을 때(가까운 후보 비교)는 출력도
    작아서 그대로 쓰면 soft하게 섞인다. sin(pi/2 * x)의 0에서의 기울기가 pi/2 (~1.57) > 1
    이라 반복 합성하면 0 근처 값이 점점 +-1로 벌어진다 (출력이 항상 [-1,1] 안에 있어서
    반복 합성해도 aliasing 걱정 없음).
    """
    x = enc_x
    for _ in range(iterations):
        x = ctx.engine.sign_bootstrap(
            x, ctx.rlk, ctx.conjugation_key, ctx.rotation_key, ctx.small_bootstrap_key
        )
    return x


def sign_argmin_compare(
    ctx, left: TournamentWinner, right: TournamentWinner, score_normalizer: float, sharpen_iterations: int = 8
) -> TournamentWinner:
    """sign_bootstrap 기반 argmin 비교. score가 작은 쪽(left)이 이기면 choose_left~1.

    score_normalizer: sign_bootstrap이 sin(pi/2 * x) 모양이라 |x|>1이면 aliasing되므로,
    (right.score - left.score)를 반드시 이 값으로 나눠서 [-1,1] 안으로 넣어야 한다
    (예: n_samples**2 같은 MGI score의 안전한 상한).
    """
    diff = ctx.engine.subtract(right.score, left.score)  # left가 작으면 양수
    normalized = ctx.engine.multiply(diff, 1.0 / score_normalizer)
    sign = sharpened_sign(ctx, normalized, sharpen_iterations)
    choose_left = ctx.engine.multiply(ctx.engine.add(sign, 1.0), 0.5)

    score = encrypted_blend(ctx, choose_left, left.score, right.score)
    threshold = encrypted_blend(ctx, choose_left, left.threshold, right.threshold)
    feature_selectors = [
        encrypted_blend(ctx, choose_left, l, r)
        for l, r in zip(left.feature_selectors, right.feature_selectors)
    ]
    left_counts = [
        encrypted_blend(ctx, choose_left, l, r) for l, r in zip(left.left_counts, right.left_counts)
    ]
    right_counts = [
        encrypted_blend(ctx, choose_left, l, r) for l, r in zip(left.right_counts, right.right_counts)
    ]
    return TournamentWinner(
        score=score,
        threshold=threshold,
        feature_selectors=feature_selectors,
        left_counts=left_counts,
        right_counts=right_counts,
    )


def train_fully_encrypted_mgi_stump(
    ctx,
    dataset: EncryptedDataset,
    candidates: list,
    score_normalizer: float | None = None,
    sharpen_iterations: int = 8,
    verbose: bool = True,
) -> TournamentWinner:
    """client decrypt 없이 depth=1 stump를 학습. 반환되는 winner는 전부 ciphertext.

    score_normalizer 기본값은 n_samples**2 (MGI score = |S|^2 - sum|S_c|^2 <= n_samples**2인
    안전한 상한).
    """
    if score_normalizer is None:
        score_normalizer = float(dataset.n_samples**2)
    winner = None
    for idx, candidate in enumerate(candidates):
        score, left_counts, right_counts = evaluate_candidate(ctx, dataset, candidate)
        state = candidate_winner_state(ctx, candidate, score, left_counts, right_counts, dataset.n_features)
        if winner is None:
            winner = state
        else:
            winner = sign_argmin_compare(
                ctx, winner, state, score_normalizer=score_normalizer, sharpen_iterations=sharpen_iterations
            )
        if verbose:
            print(f"[fully-encrypted mgi] candidate {idx + 1}/{len(candidates)} folded", flush=True)
    return winner


def debug_decrypt_winner(ctx, winner: TournamentWinner) -> dict:
    """검증 전용: winner ciphertext들을 복호화 (protocol 일부 아님)."""
    return {
        "score": float(ctx.engine.decrypt(winner.score, ctx.sk)[0].real),
        "threshold": float(ctx.engine.decrypt(winner.threshold, ctx.sk)[0].real),
        "feature_selectors": [
            float(ctx.engine.decrypt(fs, ctx.sk)[0].real) for fs in winner.feature_selectors
        ],
    }


def plaintext_mgi_best_split(X: np.ndarray, y_one_hot: np.ndarray, candidates: list):
    """검증용: plaintext로 MGI가 최소인 candidate index/threshold/score를 계산."""
    best_idx, best_score = -1, None
    for idx, candidate in enumerate(candidates):
        x = X[:, candidate.feature_idx]
        right = (x > candidate.threshold).astype(float)
        left = 1.0 - right
        left_counts = left @ y_one_hot
        right_counts = right @ y_one_hot
        score = (left_counts.sum() ** 2 - (left_counts**2).sum()) + (
            right_counts.sum() ** 2 - (right_counts**2).sum()
        )
        if best_score is None or score < best_score:
            best_score, best_idx = score, idx
    return best_idx, best_score


def _smoke_test_synthetic() -> None:
    """작은 synthetic dataset으로 정확도부터 확인 (본 iris 벤치마크 전 빠른 sanity check)."""
    rng = np.random.default_rng(0)
    n = 20
    X = np.zeros((n, 2))
    X[:10, 0] = rng.uniform(-1.0, -0.2, size=10)
    X[10:, 0] = rng.uniform(0.2, 1.0, size=10)
    X[:, 1] = rng.uniform(-1.0, 1.0, size=n)
    y = np.array([0] * 10 + [1] * 10)
    y_one_hot = one_hot_encode(y, n_classes=2)
    candidates = make_public_grid_candidates(
        n_features=2, thresholds=make_small_public_threshold_grid(candidate_count=5)
    )

    best_idx, best_score = plaintext_mgi_best_split(X, y_one_hot, candidates)
    print(f"[plaintext] best candidate={candidates[best_idx]} score={best_score:.4f}")

    ctx = create_bootstrap_context(mode="gpu")
    dataset = encrypt_dataset(ctx, X, y_one_hot)

    t0 = time.time()
    winner = train_fully_encrypted_mgi_stump(ctx, dataset, candidates)
    elapsed = time.time() - t0

    decoded = debug_decrypt_winner(ctx, winner)
    print(f"\n[fully-encrypted] elapsed={elapsed:.2f}s")
    print(f"  score decrypt={decoded['score']:.4f} (plaintext={best_score:.4f})")
    print(f"  threshold decrypt={decoded['threshold']:.4f} (plaintext={candidates[best_idx].threshold:.4f})")
    print(f"  feature_selectors decrypt={[f'{v:.4f}' for v in decoded['feature_selectors']]}")
    print(f"  expected feature_idx={candidates[best_idx].feature_idx}")


if __name__ == "__main__":
    _smoke_test_synthetic()
