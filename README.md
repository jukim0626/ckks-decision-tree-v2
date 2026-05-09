# CKKS-based Decision Tree on Iris Dataset

## 출력 결과

```
[일반 Decision Tree 정확도]: 100.0%
[테스트 샘플 수]: 30개

[일반 Decision Tree]
                  setosa  versicolor   virginica
      setosa          10           0           0
  versicolor           0           9           0
   virginica           0           0          11

CKKS 클래스 분류 중...

[CKKS Decision Tree]
                  setosa  versicolor   virginica
      setosa          10           0           0
  versicolor           0           9           0
   virginica           0           0          11

--- 결과 비교 ---
일반 DT : 100.0%
CKKS DT : 100.0%
정확도 차이 : 0.0%p
총 소요시간 : 136.55s (4.55s/sample)
```

## 구현 세부사항

- **라이브러리**: desilofhe (CKKS scheme), scikit-learn
- **데이터셋**: Iris (150개, 3 class)
- **트리 깊이**: max_depth=3
- **sigmoid 근사**: `0.5 + 0.209637x - 0.005402x^3 + 0.000050x^5`
- **CKKS 파라미터**: max_level=15, scale=2^40 (desilofhe 기본값)


## Soft Decision Tree: 확률 기반 분기 방식

### if/else vs sigmoid

일반 Decision Tree는 각 노드에서 threshold를 학습하여 다음과 같이 분기합니다.

```
x < threshold  →  왼쪽
x >= threshold →  오른쪽
```

CKKS 환경에서는 이 비교가 불가능하므로, sigmoid(x - threshold)를 사용합니다.
x - threshold의 값이 0보다 작으면 0에 수렴하고, 0보다 크면 1에 수렴하는 방식으로
if/else의 역할을 근사합니다.

---

### 확률이 리프 노드까지 분기되는 과정

각 분기 노드의 threshold 기준점이 다르기 때문에, 여러 분기점에서 sigmoid로
계산한 확률들을 weight에 할당하고, 그것들의 곱으로 리프 노드의 클래스 점수를
선택하는 방식입니다.

숫자 예시:

```
루트 노드 (petal length, threshold = 2.45)
    샘플: petal length = 1.4
    x - threshold = 1.4 - 2.45 = -1.05
    sigmoid(-1.05) ≈ 0.24  →  right_prob
    left_prob = 1 - 0.24 = 0.76

    왼쪽 자식 weight = 1.0 × 0.76 = 0.76
    오른쪽 자식 weight = 1.0 × 0.24 = 0.24

오른쪽 자식 노드 (petal width, threshold = 1.75)
    샘플: petal width = 1.3
    x - threshold = 1.3 - 1.75 = -0.45
    sigmoid(-0.45) ≈ 0.39  →  right_prob
    left_prob = 1 - 0.39 = 0.61

    왼쪽 자식 weight = 0.24 × 0.61 = 0.146  →  Versicolor 리프
    오른쪽 자식 weight = 0.24 × 0.39 = 0.094  →  Virginica 리프

최종 클래스 점수:
    Setosa(0):     0.76
    Versicolor(1): 0.146
    Virginica(2):  0.094

    argmax → 클래스 0 (Setosa) 예측 ✅
```

세 클래스 점수의 합은 항상 1.0이 되며,
argmax로 가장 높은 점수의 클래스를 최종 예측값으로 선택합니다.

---

## Encrypted Argmax: 암호화된 상태에서의 최댓값 비교

### 기존 방식의 문제

기존에는 클래스 점수 3개를 모두 복호화한 뒤 plaintext에서 argmax를 수행했습니다.
이 경우 개별 클래스 점수가 외부에 노출됩니다.

### 개선된 방식

`max(a, b) = 1/2(a+b) + 1/2·sign(a-b)` 공식을 기반으로,
비교 연산을 암호화된 상태에서 수행합니다.

**step function 근사:**
```
step(x) = (1 + sign(x)) / 2 ≈ sigmoid(SCALE · x)
```
x > 0이면 ~1, x < 0이면 ~0으로 수렴합니다. (SCALE = 5.0)

**클래스 indicator 계산:**
```
ind_i = step(si - sj) × step(si - sk)   (j, k ≠ i)
```
class i가 나머지 두 클래스를 모두 이길 때만 ~1에 수렴하며,
3개의 indicator를 복호화한 뒤 argmax로 최종 클래스를 결정합니다.

**level 예산 (max_level=15 기준):**

| 단계 | 연산 | level 소모 |
|------|------|-----------|
| 분기 노드 sigmoid | evaluate_polynomial (degree-5) | 4 |
| weight 전파 | CT × CT | 1 |
| argmax step sigmoid | multiply(scalar) + evaluate_polynomial | 5 |
| indicator 곱 | CT × CT | 1 |
| **트리 깊이 3 + argmax 합계** | | **~13** |
