"""Bootstrap 최적화 실험 전용 서브패키지.

baseline(depth1_ckks.py / depthN_ckks.py / train_depthN_ckks.py 등, 프로젝트 루트의
experiments/gradient_soft_tree/*.py)은 절대 수정하지 않는다. 이 패키지는 baseline과
같은 수학을 그대로 재현하는 것을 기본값으로 하되, ExperimentConfig 플래그로 각 최적화를
독립적으로 켜고 끌 수 있는 별도 버전을 제공한다 (OPTIMIZATION_REPORT.md Phase 0-7).
"""
