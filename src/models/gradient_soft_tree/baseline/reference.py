"""depth1_reference.py를 임의 depth로 일반화한 CKKS-매칭 plaintext 기준 구현.

**일반화 핵심 아이디어**: depth=1은 root 하나가 곧 "마지막 internal level"이라 그 gate의
dL/dgate가 바로 두 leaf의 gradient 차이(`sum_c dL_dyhat_c*(leafdist_R-leafdist_L)`)였다.
depth>1에서는 이 관계가 "leaf" 대신 "자식 노드(혹은 자식 leaf)의 reach probability에 대한
gradient(g)"로 일반화된다:

  각 노드/leaf X의 "reach probability" p_X(그 샘플이 X에 도달할 확률)에 대해
  g_X := dL/dp_X 를 정의하면:
  - leaf l: g_l = sum_c dL_dyhat_c * leafdist_l_c  (depth=1의 dL_dgate 공식과 정확히 동일한
    형태 - depth=1은 root가 곧 "마지막 레벨"이라 이 값이 바로 dL_dgate였을 뿐)
  - internal node i(자식의 reach prob에 대한 g_left, g_right를 이미 알고 있다고 하면):
      dL_dgate_i = p_i * (g_right - g_left)   (p_i = 그 노드 자신의 incoming reach prob, root는 1)
      g_i        = g_left*(1-gate_i) + g_right*gate_i   (i의 부모가 필요로 하는 값)

  dL_dgate_i가 나오면 그 다음은 depth=1과 완전히 동일한 공식으로 그 노드의
  threshold/alpha gradient를 구한다. 즉 leaf에서 root 방향으로 레벨을 거슬러 올라가며
  이 공식을 반복 적용하는 게 전부 - depth=1의 backward가 "레벨 1개짜리" 특수 케이스였을 뿐,
  본질적으로 같은 계산이 레벨마다 반복된다. (CKKS 포팅 시 이 재귀 구조를 그대로 옮기면
  depth=1에서 이미 검증된 threshold/alpha 공식을 노드마다 재사용할 수 있다.)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import load_scaled_dataset_subset, one_hot_encode  # noqa: E402
from models.gradient_soft_tree.plaintext_softmax import softmax, softmax_backward  # noqa: E402
from core.approximation.sigmoid import SIGMOID_APPROX_INTERVAL, SPLIT_SIGMOID_COEFFS, STEEPNESS  # noqa: E402
from core.encrypted_ops.softmax import _EXP_COEFFS, SOFTMAX_EXP_INTERVAL, SOFTMAX_RECIP_ITERATIONS  # noqa: E402

# 2026-10-07: approx="poly"는 CKKS가 실제로 계산하는 함수(15차 sigmoid 다항식, 20차 exp 다항식)를
# plaintext로 그대로 흉내 낸다. 이 모드와 CKKS 결과의 차이 = 순수 CKKS 노이즈, true 모드와
# poly 모드의 차이 = 다항식 근사 오차 - 두 오차를 분리해서 보고하기 위함.
# approx에 PolySpec을 직접 넘기면 다른 구간/차수의 다항식으로 흉내 낼 수 있다(ROADMAP B 탐색용).
APPROX_MODES = ("true", "poly")


@dataclass(frozen=True)
class PolySpec:
    """CKKS가 쓰는 근사 함수 묶음. 기본값 = 현재 CKKS 구현과 동일."""

    sigmoid_coeffs: tuple = tuple(SPLIT_SIGMOID_COEFFS)
    sigmoid_interval: float = SIGMOID_APPROX_INTERVAL  # x-threshold가 [-이 값, 이 값] 안이어야 유효
    exp_coeffs: tuple = tuple(_EXP_COEFFS)
    exp_interval: float = SOFTMAX_EXP_INTERVAL[1]      # softmax 입력이 [-이 값, 이 값] 안이어야 유효
    recip_iterations: int = SOFTMAX_RECIP_ITERATIONS
    steepness: float = STEEPNESS                        # sigmoid_coeffs에 baked-in된 steepness


DEFAULT_POLY_SPEC = PolySpec()


def _resolve_approx(approx) -> PolySpec | None:
    """"true" -> None(진짜 함수), "poly" -> 현재 CKKS 다항식, PolySpec -> 그대로."""
    if isinstance(approx, PolySpec):
        return approx
    if approx == "poly":
        return DEFAULT_POLY_SPEC
    if approx == "true":
        return None
    raise ValueError(f"approx must be one of {APPROX_MODES} or a PolySpec")


def _gate_values(diff: np.ndarray, steepness: float, approx) -> np.ndarray:
    """diff = x - threshold에 대한 gate. poly는 steepness=8이 계수에 이미 들어있다."""
    spec = _resolve_approx(approx)
    if spec is not None:
        if steepness != spec.steepness:
            raise ValueError(f"이 PolySpec은 steepness={spec.steepness}로 맞춘 계수다(요청: {steepness})")
        return np.polynomial.polynomial.polyval(diff, spec.sigmoid_coeffs)
    return 1.0 / (1.0 + np.exp(-steepness * diff))


def _softmax(z: np.ndarray, approx) -> np.ndarray:
    """poly는 core.encrypted_ops.softmax.packed_softmax와 같은 계산 순서를 흉내 낸다:
    exp 다항식 -> 공개 상수 exp(구간 상한)으로 나눔 -> 분모 역수를 Newton 반복(초기값 1/n)으로
    근사. 정확한 나눗셈 대신 Newton을 그대로 쓰는 이유: 분모가 작으면(모든 logit이 매우
    음수) 정해진 반복 횟수로 덜 수렴할 수 있는데, CKKS도 똑같이 덜 수렴하므로 그 효과까지
    재현해야 한다."""
    spec = _resolve_approx(approx)
    if spec is not None:
        e = np.polynomial.polynomial.polyval(z, spec.exp_coeffs) / np.exp(spec.exp_interval)
        n_valid = len(z)
        z_iter = e.sum() / n_valid
        w = e / n_valid
        for _ in range(spec.recip_iterations):
            two_minus_z = 2.0 - z_iter
            z_iter = z_iter * two_minus_z
            w = w * two_minus_z
        return w
    return softmax(z)


def init_params(n_features: int, n_classes: int, depth: int, seed: int = 0) -> dict:
    """CKKS의 init_encrypted_params_N과 같은 초기화(같은 seed, 같은 rng 호출 순서)."""
    rng = np.random.default_rng(seed)
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth
    alpha = rng.normal(0, 0.1, size=(n_internal, n_features))
    threshold = rng.normal(0, 0.1, size=(n_internal, n_features))
    leaf_logits = rng.normal(0, 0.1, size=(n_leaves, n_classes))
    return {"alpha": alpha, "threshold": threshold, "leaf_logits": leaf_logits}


def compute_gradients(
    X: np.ndarray,
    y_onehot: np.ndarray,
    params: dict,
    depth: int,
    steepness: float = 8.0,
    approx="true",
    gate_scale: float = 1.0,
    balance_lambda: float = 0.0,
    loss: str = "mse",
    sample_weight: np.ndarray | None = None,
) -> tuple[dict, float]:
    """한 iteration의 forward+backward. 반환: (grads(alpha/threshold/leaf_logits), MSE loss).

    2026-10-07: train_depthN 루프 안에 있던 계산을 그대로 뽑았다(연산/순서 동일) - optimizer를
    바꿔가며 실험하는 코드(experiments/domain_safe_optim.py)가 이 수식을 그대로 재사용하기 위함.

    2026-10-08 옵션(기본값이면 기존과 비트 단위 동일):
    - gate_scale c: gate 입력을 c*(x - t)로 줄인다. 같은 다항식으로 실효 steepness = steepness*c (c <= 1이면 구간 안전 유지)
      -> steepness continuation(처음엔 완만, 점점 날카롭게) 실험용
    - balance_lambda: 나눗셈 없는 routing balance penalty λ * Σ_i 2^-d_i * β_i^2, β_i = Σ_x P^i(x)(g_i(x) - 1/2) / n
      (Frosst-Hinton balance penalty에서 착안한 HE 친화적 변형). gradient는 dL/dg_i(x)에 2λ 2^-d β_i P^i(x)/n를 더하고,
      P^i를 통한 경로는 무시한다(stop-gradient). 반환 loss에는 포함하지 않는다(MSE만).
    - loss (2026-10-09 비교용):
      "mse"  : (1/n) Σ ||ŷ - y||² (기존)
      "wmse" : (1/n) Σ w_x ||ŷ - y||², sample_weight = 클래스 빈도 역수(평균 1) - 불균형 데이터용
      "ece"  : expected CE = -(1/n) Σ_x Σ_l μ_l(x) log q_l(y_x) (Frosst-Hinton / HME 방식). leaf gradient는
               (1/n) Σ_x μ_l(x)(q_l - y_x)로 나눗셈 없음, routing으로는 dL/dμ_l(x) = -(1/n) y_x·log q_l 전달.
               log q_l은 leaf 파라미터에만 필요(CKKS에선 범위가 정해진 log-sum-exp 다항식 근사 예정, 여기선 정확한 log)."""
    n_samples = X.shape[0]
    n_internal = (1 << depth) - 1
    n_leaves = 1 << depth
    alpha, threshold, leaf_logits = params["alpha"], params["threshold"], params["leaf_logits"]

    # ---- forward: 레벨별로 내려가며 각 노드의 gate/incoming reach prob을 저장 ----
    node_gate = [None] * n_internal
    node_gate_j = [None] * n_internal
    node_w = [None] * n_internal
    node_parent_prob = [None] * n_internal  # 그 노드 자신의 incoming reach prob (root=1)

    current_level_probs = [np.ones(n_samples)]
    node_idx = 0
    for _level in range(depth):
        next_level_probs = []
        for parent_prob_arr in current_level_probs:
            i = node_idx
            w = _softmax(alpha[i], approx)
            gate_j = _gate_values(gate_scale * (X - threshold[i]), steepness, approx)
            gate = gate_j @ w
            node_gate[i], node_gate_j[i], node_w[i] = gate, gate_j, w
            node_parent_prob[i] = parent_prob_arr
            next_level_probs.append(parent_prob_arr * (1.0 - gate))
            next_level_probs.append(parent_prob_arr * gate)
            node_idx += 1
        current_level_probs = next_level_probs
    leaf_probs = current_level_probs  # length n_leaves

    leafdist = [_softmax(leaf_logits[l], approx) for l in range(n_leaves)]
    y_hat = sum(np.outer(leaf_probs[l], leafdist[l]) for l in range(n_leaves))
    current_g = []
    dL_dleaf_logits = [None] * n_leaves
    loss_kind = loss
    if loss_kind == "ece":
        # ---- expected CE: leaf gradients + g(=dL/dμ_l) ----
        loss_val = 0.0
        for l in range(n_leaves):
            logq = np.log(np.clip(leafdist[l], 1e-12, None))
            mu = leaf_probs[l]
            dL_dleaf_logits[l] = (mu.sum() * leafdist[l] - mu @ y_onehot) / n_samples
            g_l = -(y_onehot @ logq) / n_samples
            current_g.append(g_l)
            loss_val += float((mu * (-(y_onehot @ logq))).sum() / n_samples)
        loss = loss_val
    else:
        w = np.ones(n_samples) if (loss_kind == "mse" or sample_weight is None) else np.asarray(sample_weight, dtype=float)
        loss = float((w[:, None] * (y_hat - y_onehot) ** 2).sum() / n_samples)
        dL_dyhat = (2.0 / n_samples) * (y_hat - y_onehot) * w[:, None]

        # ---- leaf gradients + g(=dL/dp_leaf) ----
        for l in range(n_leaves):
            dL_ddist_l = dL_dyhat.T @ leaf_probs[l]
            dL_dleaf_logits[l] = softmax_backward(leafdist[l], dL_ddist_l)
            current_g.append(dL_dyhat @ leafdist[l])

    # ---- 레벨을 거슬러 올라가며 노드 gradient 계산 ----
    dL_dalpha = [None] * n_internal
    dL_dthreshold = [None] * n_internal
    for level in reversed(range(depth)):
        start = (1 << level) - 1
        count = 1 << level
        next_g = [None] * count
        for idx in range(count):
            i = start + idx
            g_left, g_right = current_g[2 * idx], current_g[2 * idx + 1]
            gate, gate_j, w = node_gate[i], node_gate_j[i], node_w[i]
            p_i = node_parent_prob[i]

            dL_dgate_i = p_i * (g_right - g_left)
            next_g[idx] = g_left * (1.0 - gate) + g_right * gate
            if balance_lambda:
                beta_i = float((p_i * (gate - 0.5)).sum() / n_samples)
                dL_dgate_i = dL_dgate_i + 2.0 * balance_lambda * 2.0 ** (-level) * beta_i * p_i / n_samples

            surrogate = gate_j * (1.0 - gate_j)
            dL_dthreshold[i] = -steepness * gate_scale * w * (dL_dgate_i[:, None] * surrogate).sum(axis=0)
            term = dL_dgate_i[:, None] * (gate_j - gate[:, None])
            dL_dalpha[i] = w * term.sum(axis=0)
        current_g = next_g

    grads = {
        "alpha": np.array(dL_dalpha),
        "threshold": np.array(dL_dthreshold),
        "leaf_logits": np.array(dL_dleaf_logits),
    }
    return grads, loss


def train_depthN(
    X: np.ndarray,
    y_onehot: np.ndarray,
    depth: int,
    steepness: float = 8.0,
    lr: float = 0.3,
    epochs: int = 300,
    seed: int = 0,
    approx="true",
) -> dict:
    """vanilla full-batch GD. epochs = iteration 수(1 epoch = 파라미터 갱신 1번)."""
    _resolve_approx(approx)  # 잘못된 값이면 여기서 ValueError
    params = init_params(X.shape[1], y_onehot.shape[1], depth, seed)
    for _ in range(epochs):
        grads, _loss = compute_gradients(X, y_onehot, params, depth, steepness, approx)
        for k in ("alpha", "threshold", "leaf_logits"):
            params[k] = params[k] - lr * grads[k]
    return {**params, "steepness": steepness, "depth": depth}


def predict_proba(X: np.ndarray, params: dict, approx="true") -> np.ndarray:
    """approx="true"는 진짜 sigmoid/exp, "poly"는 CKKS와 같은 다항식(encrypted inference를
    plaintext로 흉내 낸 것 - CKKS 노이즈만 빠져 있다)."""
    _resolve_approx(approx)
    depth = params["depth"]
    steepness = params["steepness"]
    n_samples = X.shape[0]
    current_level_probs = [np.ones(n_samples)]
    node_idx = 0
    for _level in range(depth):
        next_level_probs = []
        for parent_prob_arr in current_level_probs:
            w = _softmax(params["alpha"][node_idx], approx)
            gate_j = _gate_values(X - params["threshold"][node_idx], steepness, approx)
            gate = gate_j @ w
            next_level_probs.append(parent_prob_arr * (1.0 - gate))
            next_level_probs.append(parent_prob_arr * gate)
            node_idx += 1
        current_level_probs = next_level_probs
    leaf_probs = current_level_probs
    leafdist = [_softmax(params["leaf_logits"][l], approx) for l in range(len(leaf_probs))]
    y_hat = sum(np.outer(leaf_probs[l], leafdist[l]) for l in range(len(leaf_probs)))
    return y_hat


def predict(X: np.ndarray, params: dict, approx="true") -> np.ndarray:
    """approx="true"는 진짜 sigmoid/exp, "poly"는 CKKS와 같은 다항식. params가 리스트면 multi-start 앙상블(확률 평균)."""
    if isinstance(params, (list, tuple)):
        return np.mean([predict_proba(X, p, approx) for p in params], axis=0).argmax(axis=1)
    return predict_proba(X, params, approx).argmax(axis=1)


def domain_report(X: np.ndarray, params: dict, spec: PolySpec = DEFAULT_POLY_SPEC) -> dict:
    """다항식 근사 구간 이탈 진단: sigmoid 입력 x-threshold는 [-spec.sigmoid_interval, +],
    softmax 입력(alpha, leaf_logits)은 [-spec.exp_interval, +] 안에 있어야 CKKS 결과가 의미 있다."""
    diff = X[:, None, :] - params["threshold"][None, :, :]
    out_of_range = np.abs(diff) > spec.sigmoid_interval
    max_softmax_in = max(np.abs(params["alpha"]).max(), np.abs(params["leaf_logits"]).max())
    return {
        "max_abs_gate_input": float(np.abs(diff).max()),
        "n_samples_gate_out_of_range": int(out_of_range.any(axis=(1, 2)).sum()),
        "max_abs_alpha": float(np.abs(params["alpha"]).max()),
        "max_abs_leaf_logit": float(np.abs(params["leaf_logits"]).max()),
        "softmax_out_of_range": bool(max_softmax_in > spec.exp_interval),
    }


def main():
    for dataset_name in ["iris", "wine", "breast_cancer"]:
        X_train, X_test, y_train, y_test, _ = load_scaled_dataset_subset(dataset_name)
        n_classes = int(max(y_train.max(), y_test.max()) + 1)
        y_train_oh = one_hot_encode(y_train, n_classes)
        for depth in [1, 2, 3]:
            params = train_depthN(X_train, y_train_oh, depth=depth)
            train_acc = (predict(X_train, params) == y_train).mean()
            test_acc = (predict(X_test, params) == y_test).mean()
            print(f"{dataset_name:<15} depth={depth}  train={train_acc:.4f}  test={test_acc:.4f}")


if __name__ == "__main__":
    main()
