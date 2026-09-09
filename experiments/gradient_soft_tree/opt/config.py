"""각 최적화를 독립적으로 ON/OFF할 수 있는 실험 설정.

baseline은 TreeConfig()의 기본값(모든 필드가 depthN_ckks.py/depth1_ckks.py와 동일한 동작)과
정확히 일치해야 한다 - opt/tree_ops.py의 forward_backward_update_N_opt(config=TreeConfig())가
baseline과 수치적으로 같은 결과를 내는지가 이 실험 인프라 자체의 정합성 검증 기준이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TreeConfig:
    name: str = "baseline"

    # Phase 1: softmax(attention/leaf 공용) reciprocal Newton-Raphson 반복 횟수.
    # baseline == depth1_ckks.SOFTMAX_RECIP_ITERATIONS (=10).
    reciprocal_iterations: int = 10

    # Phase 2: leaf class distribution을 softmax(probability simplex)로 만들지, 아니면
    # unconstrained score를 직접 쓸지. False면 leaf_logits ciphertext를 그대로 leaf_scores로
    # 해석해서 y_hat = sum_l leaf_prob_l * leaf_scores_l (softmax/softmax_backward 완전히 스킵).
    leaf_softmax: bool = True

    # Phase 3: 노드 feature-attention을 softmax(alpha)로 만들지, 아니면 alpha ciphertext를
    # unconstrained a_ij로 직접 쓸지 (division/exp/softmax/비교 없이 다항식 regularizer만 추가).
    attention_softmax: bool = True
    lambda_sum: float = 0.0     # R_sum = (sum_j a_ij - 1)^2
    lambda_binary: float = 0.0  # R_binary = sum_j a_ij^2 (1-a_ij)^2

    # Phase 4: gate polynomial 차수. 15 = baseline sigmoid(steepness=8) Chebyshev 근사.
    # 5 또는 3이면 opt/poly_gate.py가 생성한 low-degree "native polynomial gate" 계수를 쓴다.
    gate_degree: int = 15
    # gate poly 진입 전 level guard threshold. baseline(depthN_ckks.py 실측)은 12로
    # 고정돼있었는데, low-degree gate는 poly 자체가 레벨을 훨씬 적게 먹으므로 이 값도
    # gate_degree에 맞춰 낮춰야 최적화 의미가 있다 (opt/poly_gate.py.required_level 참고).
    gate_entry_min_level: int | None = None  # None이면 poly_gate.required_level(gate_degree)로 자동 설정
    # True면 gate의 실제 다항식 도함수 P'(z)로 backward surrogate를 쓴다 (Phase 4에서 요구하는
    # "P(x-theta) 자체가 모델"이라는 conceptual change에 정확히 대응). False면 baseline과 똑같이
    # gate_j*(1-gate_j) 로지스틱 근사를 그대로 쓴다(=degree 무관하게 baseline 수치 재현용).
    use_true_gate_derivative: bool = False

    # 그 외 ensure_level의 local min_level (baseline: depthN_ckks._LOCAL_MIN_LEVEL=5,
    # packed_softmax 내부 loop도 5). 구조적 최적화(Phase 5)용으로 노출.
    local_min_level: int = 5
    softmax_entry_min_level: int = 10  # packed_softmax의 poly 진입 전 guard (baseline 10)

    # Phase 7: optimizer. "sgd"(baseline, vanilla GD) | "momentum".
    optimizer: str = "sgd"
    momentum_beta: float = 0.9

    # 2026-08-27 발견: leaf_softmax=False(Phase 2)면 leaf score가 unconstrained라
    # attention과 마찬가지로 값 발산 위험이 있는데(attention_regularizer_grad의 R_sum
    # masking 버그로 실제 1e80 발산을 겪고 나서 발견한 리스크), leaf 쪽에는 안전장치가
    # 전혀 없었다. R_leaf = lambda_leaf * sum_c(leaf_logits_c^2)로 작은 L2 페널티를 준다 -
    # elementwise 제곱이라 packed ciphertext를 sum()할 필요가 없어 R_sum이 겪었던 종류의
    # masking 버그 자체가 구조적으로 발생할 수 없다(패딩 슬롯은 항상 0이라 제곱해도 0).
    lambda_leaf: float = 0.0

    # Phase 5A: 노드 하나의 backward per-feature loop 진입 전에 그 노드의 n_features개
    # gate_terms(threshold gradient용)와 w_j/a_j(attention gradient용)를 loop 안에서
    # 각각 개별 ensure_level로 반응적으로 처리하는 대신, merge_bootstrap으로 2개씩 묶어
    # loop 진입 전에 한 번에 새로 고친다 (Phase 0 실측: backward_threshold/backward_attention이
    # 각각 n_internal*n_features=28/28로 (node,feature) 쌍마다 거의 100% 트리거됨 - 이
    # reactive 패턴을 없애는 게 목표). False면 baseline과 동일한 reactive 방식.
    hoist_backward: bool = False

    def label(self) -> str:
        return self.name
