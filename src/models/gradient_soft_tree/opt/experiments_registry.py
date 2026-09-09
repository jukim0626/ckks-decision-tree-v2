"""이름 -> TreeConfig 매핑 (문서의 "baseline / reduced_softmax_iterations / no_leaf_softmax
/ no_attention_softmax / low_degree_gate / ..." 명명 관례를 그대로 따름). CLI에서
`python -m models.gradient_soft_tree.opt.train_opt iris 3 <preset> ...`로 고른다."""

from __future__ import annotations

from models.gradient_soft_tree.opt.config import TreeConfig

PRESETS: dict[str, TreeConfig] = {
    "baseline": TreeConfig(name="baseline"),
    # Phase 1: softmax reciprocal Newton-Raphson iteration ablation
    "recip14": TreeConfig(name="recip14", reciprocal_iterations=14),
    "recip10": TreeConfig(name="recip10", reciprocal_iterations=10),  # == baseline
    "recip8": TreeConfig(name="recip8", reciprocal_iterations=8),
    "recip6": TreeConfig(name="recip6", reciprocal_iterations=6),
    "recip4": TreeConfig(name="recip4", reciprocal_iterations=4),
    # Phase 2: leaf softmax 제거
    "no_leaf_softmax": TreeConfig(name="no_leaf_softmax", leaf_softmax=False),
    # Phase 3: attention softmax 제거 + polynomial regularizer
    "no_attn_softmax_l0": TreeConfig(name="no_attn_softmax_l0", attention_softmax=False, lambda_sum=0.0),
    "no_attn_softmax_l001": TreeConfig(name="no_attn_softmax_l001", attention_softmax=False, lambda_sum=0.001),
    "no_attn_softmax_l01": TreeConfig(name="no_attn_softmax_l01", attention_softmax=False, lambda_sum=0.01),
    "no_attn_softmax_l1": TreeConfig(name="no_attn_softmax_l1", attention_softmax=False, lambda_sum=0.1),
    # Phase 2+3 결합
    "no_leaf_no_attn": TreeConfig(
        name="no_leaf_no_attn", leaf_softmax=False, attention_softmax=False, lambda_sum=0.01
    ),
    # Phase 4: low-degree native polynomial gate (참 도함수 사용)
    "gate15_true_deriv": TreeConfig(name="gate15_true_deriv", gate_degree=15, use_true_gate_derivative=True),
    "gate5": TreeConfig(name="gate5", gate_degree=5, use_true_gate_derivative=True),
    "gate3": TreeConfig(name="gate3", gate_degree=3, use_true_gate_derivative=True),
    # Phase 5A: backward per-feature loop hoisting (merge_bootstrap 페어링), 다른 건 baseline 그대로
    "hoist_backward": TreeConfig(name="hoist_backward", hoist_backward=True),
    # Phase 2+3 (검증된 최선의 구조적 조합) + Phase 5A hoisting
    "no_leaf_no_attn_hoisted": TreeConfig(
        name="no_leaf_no_attn_hoisted", leaf_softmax=False, attention_softmax=False,
        lambda_sum=0.01, hoist_backward=True,
    ),
    # 2026-08-27: leaf_logits 발산 안전장치(lambda_leaf) 추가 버전. plaintext로
    # lambda_leaf sweep한 결과(iris depth=3, lr=0.5, 35epoch) 0.05가 최선(acc 0.9333,
    # leaf_max_abs 1.229->0.709 - 값도 억제되고 정확도도 오히려 개선됨). GPU 검증은 아직.
    "no_leaf_no_attn_safe": TreeConfig(
        name="no_leaf_no_attn_safe", leaf_softmax=False, attention_softmax=False,
        lambda_sum=0.01, lambda_leaf=0.05, hoist_backward=True,
    ),
    # 2026-08-28: hoist_backward 단독 적용이 epoch 1(-31.9%)에서는 좋아 보였지만
    # steady-state(epoch2-10, 132/epoch)에서는 baseline(113)보다 오히려 나빴던 원인 분석
    # 결과 - forward softmax(leaf/attention)와 sigmoid_forward가 steady-state에서 급격히
    # 커지는데(epoch1 0%->epoch10 100%) hoisting은 backward만 건드려서 그쪽을 못 잡음.
    # softmax는 그대로 두되(정확도/lr 리스크 없음) Phase1(reciprocal 축소, forward softmax
    # 비용 감소)+Phase4(저차수 gate, sigmoid_forward guard 임계값 하향)+hoisting을 같이
    # 적용해서 steady-state 전체를 커버하는 조합. leaf/attention softmax는 안 건드리므로
    # lr=2.0 그대로 써도 안전(발산 리스크 없음).
    "safe_combined": TreeConfig(
        name="safe_combined",
        reciprocal_iterations=6,
        gate_degree=5,
        use_true_gate_derivative=True,
        hoist_backward=True,
    ),
    # F. combined FHE-friendly version (Phase 1+2+3+4를 안전 마진 있게 결합)
    "combined": TreeConfig(
        name="combined",
        reciprocal_iterations=8,
        leaf_softmax=False,
        attention_softmax=False,
        lambda_sum=0.01,
        gate_degree=5,
        use_true_gate_derivative=True,
    ),
}


def get_preset(name: str) -> TreeConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}. available: {sorted(PRESETS)}")
    return PRESETS[name]
