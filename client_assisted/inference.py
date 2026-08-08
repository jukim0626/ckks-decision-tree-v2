"""Depth와 무관한 encrypted inference 공용 helper.

새 sample encrypt, 최종 class_scores decrypt+argmax는 어떤 depth/모델 구조를 쓰든
동일하다 (client가 아는 유일한 예외적 decrypt 지점). encrypted_depth2_inference.py와
encrypted_fixed_depth_inference.py가 둘 다 이 함수들을 재사용한다.
"""

from __future__ import annotations

import numpy as np

from client_assisted.client_ops import client_decrypt_scalar
from client_assisted.context import EncryptedTrainingContext


def encrypt_inference_sample(
    ctx: EncryptedTrainingContext,
    sample: np.ndarray,
    pk,
) -> list:
    """새 sample의 feature 각각을 single-slot ciphertext로 encrypt.

    ckks_tree.py의 predict_ckks가 하던 것과 동일한 패턴이다 (slot 0에만 실제
    값을 넣고 나머지 slot은 0으로 채워짐). 학습 때 만든 encrypted split(feature
    mask/threshold)은 n_samples 길이 constant vector이지만, CKKS 연산은 항상
    slot-wise이고 최종적으로 slot 0만 읽으므로 길이를 맞출 필요가 없다.
    """
    return [
        ctx.engine.encrypt([float(sample[feature_idx])], pk)
        for feature_idx in range(len(sample))
    ]


def encrypted_predict_class(
    ctx: EncryptedTrainingContext,
    class_scores: list,
    sk,
) -> int:
    """논문 방식: server가 만든 encrypted class_scores를 client가 그대로 decrypt해서
    plaintext argmax (유일한 decrypt 예외). ckks_tree.py의 encrypted_argmax(pairwise
    compare-sigmoid)는 논문이 요구하지 않는 별도 privacy 속성이라 여기서는 쓰지 않는다."""
    decrypted_scores = [float(ctx.engine.decrypt(score, sk)[0]) for score in class_scores]
    return int(np.argmax(decrypted_scores))


def debug_decrypt_class_scores(
    ctx: EncryptedTrainingContext,
    class_scores: list,
) -> list[float]:
    """검증용. encrypted_traverse_and_predict* / encrypted_predict_class 내부에서는 호출 안 함."""
    return [client_decrypt_scalar(ctx, score) for score in class_scores]
