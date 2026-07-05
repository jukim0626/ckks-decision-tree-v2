# Pima Indians Diabetes Decision Tree

이 폴더는 Pima Indians Diabetes dataset으로 깊은 Decision Tree를 학습하고,
학습된 if/else 분기를 모두 Markdown으로 저장한 뒤 DESILO FHE 기반 soft
Decision Tree 추론까지 실행하는 실험 코드입니다.

## Dataset

- 샘플 수: 768
- feature 수: 8
- label: binary
  - `0`: normal
  - `1`: diabetes

## Features

| feature | 해석 예시 |
|---|---|
| `pregnancies` | 임신 횟수 |
| `glucose` | glucose > 140이면 고혈당 위험 |
| `blood_pressure` | blood_pressure > 80이면 고혈압 위험 |
| `skin_thickness` | 피부 두께 |
| `insulin` | insulin > 200이면 비정상 위험 |
| `bmi` | BMI > 30이면 비만 위험 |
| `diabetes_pedigree` | 가족력 기반 diabetes pedigree function |
| `age` | age > 45이면 위험 연령 |

## 실행

repo root에서 실행합니다.

```bash
source .venv/bin/activate
python -m pima_diabetes.main
```

처음 실행할 때 `pima-indians-diabetes.csv`가 없으면 공개 CSV URL에서 자동으로
다운로드합니다.

## 실행 모드

M1 MacBook Air에서 깊은 Decision Tree의 전체 FHE 추론은 오래 걸릴 수 있습니다.
그래서 plain Decision Tree와 전체 if/else rule 저장은 깊게 수행하고, FHE는
샘플 수를 조절해서 검증합니다.

### 깊은 plain Decision Tree + 전체 rule 저장

```bash
python -m pima_diabetes.main --max-depth 20
```

현재 데이터 split 기준 결과:

```text
[일반 Decision Tree 정확도]: 72.7%
[테스트 샘플 수]: 154개
[트리 max_depth 설정]: 20
[학습된 트리 실제 depth]: 13
[학습된 트리 leaf 수]: 96
```

### M1용 FHE smoke test

```bash
python -m pima_diabetes.main --max-depth 3 --run-fhe --fhe-samples 3 --fhe-max-level 15
```

검증된 실행 결과:

```text
FHE DT subset     : 100.0% (3 samples)
총 소요시간 : 19.79s (6.60s/sample)
```

조금 더 깊은 depth 5도 M1에서 실행은 가능하지만 매우 느립니다.

```bash
python -m pima_diabetes.main --max-depth 5 --run-fhe --fhe-samples 2 --fhe-max-level 20
```

검증된 실행 결과:

```text
FHE DT subset     : 100.0% (2 samples)
총 소요시간 : 118.53s (59.26s/sample)
```

### 연구실 서버용 더 깊은 FHE test

```bash
python -m pima_diabetes.main --max-depth 10 --run-fhe --fhe-samples 154 --fhe-max-level 28
```

## 출력 파일

- `tree_rules_max_depth_20_actual_depth_13.md`: 깊은 Decision Tree의 모든 if/else 분기
- `tree_rules_max_depth_5_actual_depth_5.md`: M1에서 검증한 중간 깊이 FHE test용 분기
- `tree_rules_max_depth_3_actual_depth_3.md`: M1 FHE smoke test용 얕은 tree 분기

## 구현 메모

- Pima dataset은 `sklearn.datasets`에 내장되어 있지 않습니다.
- `glucose`, `blood_pressure`, `skin_thickness`, `insulin`, `bmi`의 0 값은
  missing value로 보고 train split의 median으로 imputation합니다.
- Decision Tree 학습과 FHE 추론은 CKKS polynomial sigmoid 안정성을 위해
  standardized feature에서 수행합니다.
- `tree_rules.md`에는 사람이 해석하기 쉽도록 raw feature 단위 threshold도 함께
  출력합니다.
