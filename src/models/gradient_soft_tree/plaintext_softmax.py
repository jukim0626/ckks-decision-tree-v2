"""gradient_soft_tree의 모든 plaintext reference 구현(baseline/local_loss/opt의 실험
변형들)이 공유하는 순수 plaintext softmax + backward. CKKS와는 무관한 numpy 전용 코드.

2026-09-09: depth1_reference.py(원래 depth=1 전용 파일이었는데, depthN_reference.py가
depth=1을 포함해 일반화하면서 이 두 함수만 계속 여기저기서 재사용되고 있었음)에서 분리."""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def softmax_backward(dist: np.ndarray, dL_ddist: np.ndarray) -> np.ndarray:
    """d(softmax)/d(logit) 야코비안을 적용: dL/dlogit_k = dist_k*(dL_ddist_k - sum_c dL_ddist_c*dist_c)."""
    s = float(np.dot(dL_ddist, dist))
    return dist * (dL_ddist - s)
