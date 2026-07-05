"""Fully encrypted MGII training prototype for a Decision Stump.

이 파일은 depth=1 stump만 다룬다. training core는 plaintext 비교/argmin/decrypt를
사용하지 않고, ciphertext arithmetic으로 MGII score와 encrypted selector를 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from desilofhe import Engine

from ckks_tree import COMPARE_SCALE, sigmoid_approx_enc


@dataclass(frozen=True)
class PublicSplitCandidate:
    """공개 candidate split 정보."""

    feature_idx: int
    threshold: float


@dataclass
class EncryptedTrainingContext:
    """DESILO FHE encrypted training에 필요한 key/context 묶음."""

    engine: Engine
    sk: object
    pk: object
    rlk: object
    rotation_key: object
    mode: str
    device_id: int


@dataclass
class EncryptedDataset:
    """slot packing된 encrypted training data."""

    enc_features: list
    enc_labels: list
    n_samples: int
    n_classes: int


@dataclass
class EncryptedSplitEvaluation:
    """candidate 하나에 대한 encrypted MGII 평가 결과."""

    candidate: PublicSplitCandidate
    left_prob: object
    right_prob: object
    left_counts: list
    right_counts: list
    mgii_score: object


@dataclass
class EncryptedStumpModel:
    """encrypted MGII stump training 결과."""

    candidates: list[PublicSplitCandidate]
    evaluations: list[EncryptedSplitEvaluation]
    selectors: list
    selected_threshold: object
    feature_selectors: list
    left_leaf_counts: list
    right_leaf_counts: list


@dataclass
class EncryptedTournamentWinner:
    """encrypted tournament argmin의 현재 winner 상태."""

    score: object
    threshold: object
    feature_selectors: list
    left_counts: list
    right_counts: list


def create_encrypted_training_context(
    mode: str = "gpu",
    device_id: int = 0,
    max_level: int = 30,
) -> EncryptedTrainingContext:
    """encrypted MGII training용 DESILO FHE context를 생성."""
    engine = Engine(max_level=max_level, mode=mode, device_id=device_id)
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


def encrypt_training_data(
    ctx: EncryptedTrainingContext,
    X: np.ndarray,
    y_one_hot: np.ndarray,
) -> EncryptedDataset:
    """client-side plaintext data를 slot packing한 ciphertext로 암호화."""
    enc_features = [
        ctx.engine.encrypt(X[:, feature_idx].astype(float).tolist(), ctx.pk)
        for feature_idx in range(X.shape[1])
    ]
    enc_labels = [
        ctx.engine.encrypt(y_one_hot[:, class_idx].astype(float).tolist(), ctx.pk)
        for class_idx in range(y_one_hot.shape[1])
    ]
    return EncryptedDataset(
        enc_features=enc_features,
        enc_labels=enc_labels,
        n_samples=X.shape[0],
        n_classes=y_one_hot.shape[1],
    )


def make_public_grid_candidates(
    n_features: int,
    thresholds: np.ndarray,
) -> list[PublicSplitCandidate]:
    """public threshold grid로 split 후보 목록을 생성."""
    candidates = []
    for feature_idx in range(n_features):
        for threshold in thresholds:
            candidates.append(
                PublicSplitCandidate(
                    feature_idx=feature_idx,
                    threshold=float(threshold),
                )
            )
    return candidates


def make_small_public_threshold_grid(
    candidate_count: int = 3,
    feature_range: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """암호화 실험용으로 작은 public threshold grid를 생성."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be at least 1")
    low, high = feature_range
    if candidate_count == 1:
        return np.array([(low + high) / 2.0], dtype=float)
    return np.linspace(low, high, candidate_count + 2, dtype=float)[1:-1]


def encrypted_slot_sum(ctx: EncryptedTrainingContext, enc_values) -> object:
    """ciphertext slot 전체 합을 ciphertext로 계산."""
    return ctx.engine.sum(enc_values, ctx.rotation_key)


def encrypted_class_counts(
    ctx: EncryptedTrainingContext,
    enc_weights,
    enc_labels: list,
) -> list:
    """encrypted sample weights와 encrypted one-hot label로 클래스별 count 계산."""
    counts = []
    for enc_label in enc_labels:
        weighted_label = ctx.engine.multiply(enc_weights, enc_label, ctx.rlk)
        counts.append(encrypted_slot_sum(ctx, weighted_label))
    return counts


def encrypted_mgii_from_counts(
    ctx: EncryptedTrainingContext,
    class_counts: list,
) -> object:
    """encrypted class count로 |D|^2 - sum_i |D_i|^2 계산."""
    total_count = class_counts[0]
    for count in class_counts[1:]:
        total_count = ctx.engine.add(total_count, count)

    squared_total = ctx.engine.multiply(total_count, total_count, ctx.rlk)
    squared_class_sum = ctx.engine.multiply(class_counts[0], class_counts[0], ctx.rlk)
    for count in class_counts[1:]:
        squared_count = ctx.engine.multiply(count, count, ctx.rlk)
        squared_class_sum = ctx.engine.add(squared_class_sum, squared_count)

    return ctx.engine.subtract(squared_total, squared_class_sum)


def evaluate_encrypted_candidate(
    ctx: EncryptedTrainingContext,
    dataset: EncryptedDataset,
    candidate: PublicSplitCandidate,
) -> EncryptedSplitEvaluation:
    """candidate 하나에 대해 encrypted comparison, aggregate, MGII를 계산."""
    enc_feature = dataset.enc_features[candidate.feature_idx]
    enc_diff = ctx.engine.subtract(enc_feature, candidate.threshold)
    enc_compare_input = ctx.engine.multiply(enc_diff, COMPARE_SCALE)
    right_prob = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_compare_input)
    left_prob = ctx.engine.subtract(1.0, right_prob)

    left_counts = encrypted_class_counts(ctx, left_prob, dataset.enc_labels)
    right_counts = encrypted_class_counts(ctx, right_prob, dataset.enc_labels)
    left_mgii = encrypted_mgii_from_counts(ctx, left_counts)
    right_mgii = encrypted_mgii_from_counts(ctx, right_counts)
    mgii_score = ctx.engine.add(left_mgii, right_mgii)

    return EncryptedSplitEvaluation(
        candidate=candidate,
        left_prob=left_prob,
        right_prob=right_prob,
        left_counts=left_counts,
        right_counts=right_counts,
        mgii_score=mgii_score,
    )


def encrypted_argmin_selectors(
    ctx: EncryptedTrainingContext,
    scores: list,
    compare_scale: float,
    score_normalizer: float,
) -> list:
    """encrypted MGII score list에서 argmin selector를 polynomial 비교로 근사."""
    selectors = []
    n_scores = len(scores)
    scale_factor = compare_scale / score_normalizer
    for i in range(n_scores):
        selector = None
        for j in range(n_scores):
            if i == j:
                continue
            diff = ctx.engine.subtract(scores[j], scores[i])
            scaled_diff = ctx.engine.multiply(diff, scale_factor)
            i_better_than_j = sigmoid_approx_enc(ctx.engine, ctx.rlk, scaled_diff)
            if selector is None:
                selector = i_better_than_j
            else:
                selector = ctx.engine.multiply(selector, i_better_than_j, ctx.rlk)
        if selector is None:
            selector = ctx.engine.encrypt([1.0], ctx.pk)
        selectors.append(selector)
    return selectors


def encrypted_select_counts(
    ctx: EncryptedTrainingContext,
    selectors: list,
    candidate_counts: list[list],
    n_classes: int,
) -> list:
    """encrypted selector로 candidate별 class count를 weighted sum."""
    selected_counts = []
    for class_idx in range(n_classes):
        selected_count = None
        for selector, counts in zip(selectors, candidate_counts):
            selected_piece = ctx.engine.multiply(selector, counts[class_idx], ctx.rlk)
            if selected_count is None:
                selected_count = selected_piece
            else:
                selected_count = ctx.engine.add(selected_count, selected_piece)
        selected_counts.append(selected_count)
    return selected_counts


def encrypted_select_threshold(
    ctx: EncryptedTrainingContext,
    selectors: list,
    candidates: list[PublicSplitCandidate],
) -> object:
    """encrypted selector로 selected threshold를 ciphertext 상태로 계산."""
    selected_threshold = None
    for selector, candidate in zip(selectors, candidates):
        selected_piece = ctx.engine.multiply(selector, candidate.threshold)
        if selected_threshold is None:
            selected_threshold = selected_piece
        else:
            selected_threshold = ctx.engine.add(selected_threshold, selected_piece)
    return selected_threshold


def encrypted_feature_selectors(
    ctx: EncryptedTrainingContext,
    selectors: list,
    candidates: list[PublicSplitCandidate],
    n_features: int,
) -> list:
    """각 feature가 선택됐는지를 encrypted mask 형태로 계산."""
    feature_masks = []
    for feature_idx in range(n_features):
        feature_mask = None
        for selector, candidate in zip(selectors, candidates):
            if candidate.feature_idx != feature_idx:
                continue
            if feature_mask is None:
                feature_mask = selector
            else:
                feature_mask = ctx.engine.add(feature_mask, selector)
        if feature_mask is None:
            feature_mask = ctx.engine.encrypt([0.0], ctx.pk)
        feature_masks.append(feature_mask)
    return feature_masks


def encrypted_selected_feature_value(
    ctx: EncryptedTrainingContext,
    enc_features: list,
    feature_selectors: list,
) -> object:
    """encrypted feature mask로 선택된 feature value를 ciphertext로 계산."""
    selected_feature = None
    for enc_feature, feature_selector in zip(enc_features, feature_selectors):
        selected_piece = ctx.engine.multiply(enc_feature, feature_selector, ctx.rlk)
        if selected_feature is None:
            selected_feature = selected_piece
        else:
            selected_feature = ctx.engine.add(selected_feature, selected_piece)
    return selected_feature


def encrypted_candidate_winner_state(
    ctx: EncryptedTrainingContext,
    evaluation: EncryptedSplitEvaluation,
    n_features: int,
) -> EncryptedTournamentWinner:
    """candidate evaluation을 tournament winner 상태로 변환."""
    feature_selectors = []
    for feature_idx in range(n_features):
        value = 1.0 if feature_idx == evaluation.candidate.feature_idx else 0.0
        feature_selectors.append(ctx.engine.encrypt([value], ctx.pk))
    return EncryptedTournamentWinner(
        score=evaluation.mgii_score,
        threshold=ctx.engine.encrypt([evaluation.candidate.threshold], ctx.pk),
        feature_selectors=feature_selectors,
        left_counts=evaluation.left_counts,
        right_counts=evaluation.right_counts,
    )


def encrypted_blend(
    ctx: EncryptedTrainingContext,
    choose_left,
    left_value,
    right_value,
) -> object:
    """choose_left가 1에 가까우면 left_value, 0에 가까우면 right_value 선택."""
    left_piece = ctx.engine.multiply(choose_left, left_value, ctx.rlk)
    right_weight = ctx.engine.subtract(1.0, choose_left)
    right_piece = ctx.engine.multiply(right_weight, right_value, ctx.rlk)
    return ctx.engine.add(left_piece, right_piece)


def encrypted_tournament_compare(
    ctx: EncryptedTrainingContext,
    left: EncryptedTournamentWinner,
    right: EncryptedTournamentWinner,
    compare_scale: float,
    score_normalizer: float,
) -> EncryptedTournamentWinner:
    """두 encrypted winner 후보 중 MGII score가 작은 쪽을 encrypted로 선택."""
    score_diff = ctx.engine.subtract(right.score, left.score)
    scaled_diff = ctx.engine.multiply(score_diff, compare_scale / score_normalizer)
    choose_left = sigmoid_approx_enc(ctx.engine, ctx.rlk, scaled_diff)

    score = encrypted_blend(ctx, choose_left, left.score, right.score)
    threshold = encrypted_blend(ctx, choose_left, left.threshold, right.threshold)
    feature_selectors = [
        encrypted_blend(ctx, choose_left, left_mask, right_mask)
        for left_mask, right_mask in zip(left.feature_selectors, right.feature_selectors)
    ]
    left_counts = [
        encrypted_blend(ctx, choose_left, left_count, right_count)
        for left_count, right_count in zip(left.left_counts, right.left_counts)
    ]
    right_counts = [
        encrypted_blend(ctx, choose_left, left_count, right_count)
        for left_count, right_count in zip(left.right_counts, right.right_counts)
    ]
    return EncryptedTournamentWinner(
        score=score,
        threshold=threshold,
        feature_selectors=feature_selectors,
        left_counts=left_counts,
        right_counts=right_counts,
    )


def encrypted_tournament_argmin(
    ctx: EncryptedTrainingContext,
    evaluations: list[EncryptedSplitEvaluation],
    n_features: int,
    score_normalizer: float,
    compare_scale: float = 30.0,
) -> EncryptedTournamentWinner:
    """candidate들을 tournament 방식으로 비교해 encrypted winner를 계산."""
    if not evaluations:
        raise ValueError("evaluations must not be empty")

    round_items = [
        encrypted_candidate_winner_state(ctx, evaluation, n_features)
        for evaluation in evaluations
    ]
    while len(round_items) > 1:
        next_round = []
        for idx in range(0, len(round_items), 2):
            if idx + 1 >= len(round_items):
                next_round.append(round_items[idx])
                continue
            winner = encrypted_tournament_compare(
                ctx=ctx,
                left=round_items[idx],
                right=round_items[idx + 1],
                compare_scale=compare_scale,
                score_normalizer=score_normalizer,
            )
            next_round.append(winner)
        round_items = next_round
    return round_items[0]


def train_encrypted_mgii_stump(
    ctx: EncryptedTrainingContext,
    dataset: EncryptedDataset,
    candidates: list[PublicSplitCandidate],
    n_features: int,
    argmin_compare_scale: float = 30.0,
    selection_method: str = "tournament",
) -> EncryptedStumpModel:
    """fully encrypted MGII 방식으로 depth=1 stump를 학습."""
    evaluations = [
        evaluate_encrypted_candidate(ctx, dataset, candidate)
        for candidate in candidates
    ]
    if selection_method == "pairwise":
        scores = [evaluation.mgii_score for evaluation in evaluations]
        selectors = encrypted_argmin_selectors(
            ctx,
            scores,
            compare_scale=argmin_compare_scale,
            score_normalizer=float(dataset.n_samples * dataset.n_samples),
        )
        selected_threshold = encrypted_select_threshold(
            ctx=ctx,
            selectors=selectors,
            candidates=candidates,
        )
        feature_selectors = encrypted_feature_selectors(
            ctx=ctx,
            selectors=selectors,
            candidates=candidates,
            n_features=n_features,
        )
        left_leaf_counts = encrypted_select_counts(
            ctx=ctx,
            selectors=selectors,
            candidate_counts=[evaluation.left_counts for evaluation in evaluations],
            n_classes=dataset.n_classes,
        )
        right_leaf_counts = encrypted_select_counts(
            ctx=ctx,
            selectors=selectors,
            candidate_counts=[evaluation.right_counts for evaluation in evaluations],
            n_classes=dataset.n_classes,
        )
    elif selection_method == "tournament":
        winner = encrypted_tournament_argmin(
            ctx=ctx,
            evaluations=evaluations,
            n_features=n_features,
            score_normalizer=float(dataset.n_samples * dataset.n_samples),
            compare_scale=argmin_compare_scale,
        )
        selectors = []
        selected_threshold = winner.threshold
        feature_selectors = winner.feature_selectors
        left_leaf_counts = winner.left_counts
        right_leaf_counts = winner.right_counts
    else:
        raise ValueError("selection_method must be 'tournament' or 'pairwise'")

    return EncryptedStumpModel(
        candidates=candidates,
        evaluations=evaluations,
        selectors=selectors,
        selected_threshold=selected_threshold,
        feature_selectors=feature_selectors,
        left_leaf_counts=left_leaf_counts,
        right_leaf_counts=right_leaf_counts,
    )


def encrypt_sample(
    ctx: EncryptedTrainingContext,
    sample: np.ndarray,
) -> list:
    """client-side sample feature를 inference용 ciphertext list로 암호화."""
    return [
        ctx.engine.encrypt([float(sample[feature_idx])], ctx.pk)
        for feature_idx in range(len(sample))
    ]


def encrypted_stump_predict_scores(
    ctx: EncryptedTrainingContext,
    model: EncryptedStumpModel,
    enc_sample_features: list,
) -> list:
    """encrypted stump model과 encrypted sample로 encrypted class score 계산."""
    selected_feature = encrypted_selected_feature_value(
        ctx=ctx,
        enc_features=enc_sample_features,
        feature_selectors=model.feature_selectors,
    )
    enc_diff = ctx.engine.subtract(selected_feature, model.selected_threshold)
    enc_compare_input = ctx.engine.multiply(enc_diff, COMPARE_SCALE)
    right_prob = sigmoid_approx_enc(ctx.engine, ctx.rlk, enc_compare_input)
    left_prob = ctx.engine.subtract(1.0, right_prob)

    class_scores = []
    for left_count, right_count in zip(model.left_leaf_counts, model.right_leaf_counts):
        left_score = ctx.engine.multiply(left_prob, left_count, ctx.rlk)
        right_score = ctx.engine.multiply(right_prob, right_count, ctx.rlk)
        class_scores.append(ctx.engine.add(left_score, right_score))
    return class_scores


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    """client-side label을 암호화 전 one-hot matrix로 변환."""
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y] = 1.0
    return y_one_hot


def client_decrypt_vector(
    ctx: EncryptedTrainingContext,
    ciphertext,
    length: int,
) -> np.ndarray:
    """검증용 client-side decrypt helper. encrypted training core에서는 사용하지 않음."""
    return np.array(ctx.engine.decrypt(ciphertext, ctx.sk)[:length])


def client_decrypt_scalar(
    ctx: EncryptedTrainingContext,
    ciphertext,
) -> float:
    """검증용 client-side scalar decrypt helper."""
    return float(ctx.engine.decrypt(ciphertext, ctx.sk)[0])


def sigmoid_approx_plain(x: np.ndarray) -> np.ndarray:
    """검증용 plaintext sigmoid polynomial."""
    from sigmoid_approx_coeffs import SIGMOID_COEFFS

    return np.polynomial.polynomial.polyval(x, SIGMOID_COEFFS)


def plaintext_mgii_debug(
    X: np.ndarray,
    y_one_hot: np.ndarray,
    candidates: list[PublicSplitCandidate],
) -> tuple[list[float], int]:
    """검증용 plaintext MGII score와 best candidate index 계산."""
    scores = []
    for candidate in candidates:
        compare_input = COMPARE_SCALE * (X[:, candidate.feature_idx] - candidate.threshold)
        right_prob = sigmoid_approx_plain(compare_input)
        left_prob = 1.0 - right_prob

        left_counts = np.sum(left_prob[:, None] * y_one_hot, axis=0)
        right_counts = np.sum(right_prob[:, None] * y_one_hot, axis=0)
        left_total = np.sum(left_counts)
        right_total = np.sum(right_counts)
        left_mgii = left_total * left_total - np.sum(left_counts * left_counts)
        right_mgii = right_total * right_total - np.sum(right_counts * right_counts)
        scores.append(float(left_mgii + right_mgii))

    return scores, int(np.argmin(scores))


def debug_decrypt_stump(
    ctx: EncryptedTrainingContext,
    model: EncryptedStumpModel,
    X: np.ndarray,
    y_one_hot: np.ndarray,
    class_scores: list,
) -> None:
    """client-side decrypt로 encrypted stump 결과를 검산."""
    plaintext_scores, expected_idx = plaintext_mgii_debug(
        X=X,
        y_one_hot=y_one_hot,
        candidates=model.candidates,
    )
    encrypted_scores = [
        client_decrypt_scalar(ctx, evaluation.mgii_score)
        for evaluation in model.evaluations
    ]
    selectors = [
        client_decrypt_scalar(ctx, selector)
        for selector in model.selectors
    ]
    feature_selectors = [
        client_decrypt_scalar(ctx, feature_selector)
        for feature_selector in model.feature_selectors
    ]
    selected_threshold = client_decrypt_scalar(ctx, model.selected_threshold)
    decrypted_class_scores = [
        client_decrypt_scalar(ctx, score)
        for score in class_scores
    ]

    print("\n[client debug decrypt]")
    print("candidate scores:")
    for idx, candidate in enumerate(model.candidates):
        line = (
            f"  {idx}: feature={candidate.feature_idx} | "
            f"threshold={candidate.threshold:.6f} | "
            f"plain_mgii={plaintext_scores[idx]:.6f} | "
            f"enc_mgii={encrypted_scores[idx]:.6f}"
        )
        if selectors:
            line += f" | selector={selectors[idx]:.6f}"
        print(line)
    print(f"expected plaintext best index: {expected_idx}")
    print(f"selected threshold decrypt: {selected_threshold:.6f}")
    print(f"feature selector decrypt: {[round(v, 6) for v in feature_selectors]}")
    print(f"prediction score decrypt: {[round(v, 6) for v in decrypted_class_scores]}")


def sync_engine_if_needed(ctx: EncryptedTrainingContext) -> None:
    """DESILO FHE async GPU mode에서만 engine 작업 완료를 기다림."""
    try:
        ctx.engine.sync()
    except RuntimeError as exc:
        if "Async GPU mode" not in str(exc):
            raise


def smoke_test() -> None:
    """작은 synthetic dataset으로 encrypted stump training pipeline을 실행."""
    X = np.array(
        [
            [-0.8, -0.1],
            [-0.5, 0.2],
            [0.4, -0.2],
            [0.7, 0.1],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1], dtype=int)
    y_one_hot = one_hot_encode(y, n_classes=2)
    candidates = make_public_grid_candidates(
        n_features=X.shape[1],
        thresholds=make_small_public_threshold_grid(candidate_count=2),
    )

    ctx = create_encrypted_training_context(mode="gpu", max_level=30)
    dataset = encrypt_training_data(ctx, X, y_one_hot)
    model = train_encrypted_mgii_stump(
        ctx=ctx,
        dataset=dataset,
        candidates=candidates,
        n_features=X.shape[1],
        selection_method="tournament",
    )
    enc_sample_features = encrypt_sample(ctx, X[0])
    class_scores = encrypted_stump_predict_scores(ctx, model, enc_sample_features)
    sync_engine_if_needed(ctx)

    print("[encrypted MGII stump smoke test]")
    print(f"mode={ctx.mode} | candidates={len(candidates)} | samples={dataset.n_samples}")
    print("selection_method=tournament")
    print("training core completed without plaintext argmin/decrypt")
    print(f"pairwise selectors stored: {len(model.selectors)}")
    print("encrypted selected threshold: ready")
    print(f"encrypted feature selectors: {len(model.feature_selectors)}")
    print(f"encrypted left leaf counts: {len(model.left_leaf_counts)}")
    print(f"encrypted right leaf counts: {len(model.right_leaf_counts)}")
    print(f"encrypted prediction scores: {len(class_scores)}")
    debug_decrypt_stump(ctx, model, X, y_one_hot, class_scores)


if __name__ == "__main__":
    smoke_test()
