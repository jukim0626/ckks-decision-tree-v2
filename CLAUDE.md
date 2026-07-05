# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# 가상환경 활성화 (항상 먼저 실행)
source .venv/bin/activate

# 메인 실험 실행 (일반 DT vs CKKS DT 정확도 비교)
python main.py

# sigmoid 근사 계수 재계산 (Chebyshev 근사)
python sigmoid_approx_coeffs.py
```

## Architecture

이 프로젝트는 CKKS 동형 암호 기반 Decision Tree 추론을 구현합니다. 데이터를 암호화한 상태에서 복호화 없이 분류를 수행합니다.

### 파일 역할

| 파일 | 역할 |
|------|------|
| `train_tree.py` | scikit-learn으로 Decision Tree 학습 후 트리 구조(임계값, 분기 인덱스 등)를 dict로 추출. `dataset_name`으로 iris/wine/breast_cancer 선택 가능 |
| `ckks_tree.py` | desilofhe CKKS로 암호화 추론 수행 (n-class 일반화) |
| `sigmoid_approx_coeffs.py` | Chebyshev 근사로 sigmoid 5차 다항식 계수를 도출하는 유틸리티 |
| `main.py` | 일반 DT와 CKKS DT 결과를 비교하는 엔트리포인트 |

### 핵심 설계: Soft Decision Tree

CKKS 암호문에서는 `x < threshold` 같은 비교 연산이 불가능합니다. 이를 우회하기 위해 각 분기 노드에서 sigmoid를 사용합니다.

```
right_prob = sigmoid(x - threshold)   # 5차 다항식 근사
left_prob  = 1 - right_prob
```

루트부터 리프까지 각 분기의 확률을 **곱(weight 전파)** 해가며 최종 클래스 점수를 계산하고, argmax로 예측합니다. 클래스 점수의 합은 항상 1.0.

### CKKS 파라미터 (ckks_tree.py)

- `poly_modulus_degree`: 32768
- `coeff_mod_bit_sizes`: `[60, 40, 40, 40, 40, 40, 40, 40, 60]` (depth 7)
- `global_scale`: 2^40
- sigmoid 근사 1회당 depth 3 소모, weight 전파 1회당 depth 1 소모

### sigmoid 근사 다항식 (ckks_tree.py)

```
sigmoid(x) ≈ 0.5 + 0.209637x - 0.005402x³ + 0.000050x⁵
```

계수를 변경하려면 `sigmoid_approx_coeffs.py`를 실행해 새 계수를 구한 뒤 `ckks_tree.py`의 `sigmoid_approx_enc` 함수에 반영합니다.

### 의존성

- `tenseal==0.3.16` — CKKS 동형 암호 연산
- `scikit-learn==1.8.0` — Decision Tree 학습
- `numpy==2.4.4`
