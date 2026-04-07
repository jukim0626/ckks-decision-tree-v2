# CKKS-based Decision Tree on Iris Dataset

## 출력 결과

```
[일반 Decision Tree 정확도]: 100.0%
[테스트 샘플 수]: 30개

[일반 Decision Tree]
              setosa  versicolor   virginica
      setosa      10           0           0
  versicolor       0           9           0
   virginica       0           0          11
소요시간 : 0.0003s (0.0000s/sample)

CKKS context 생성 중...
분류 시작

[CKKS Decision Tree]
              setosa  versicolor   virginica
      setosa      10           0           0
  versicolor       0           9           0
   virginica       0           0          11
소요시간 : 12.17s (0.41s/sample)

--- 결과 비교 ---
일반 DT   : 100.0%
CKKS DT   : 100.0%
정확도 차이 : 0.0%p
```

## 구현 세부사항

- **라이브러리**: TenSEAL (CKKS scheme), scikit-learn
- **데이터셋**: Iris (150개, 3 class)
- **트리 깊이**: max_depth=3
- **sigmoid 근사**: `0.5 + 0.197x - 0.004x³` (3차 다항식)
- **CKKS 파라미터**: poly_modulus_degree=32768, depth=6, scale=2^40