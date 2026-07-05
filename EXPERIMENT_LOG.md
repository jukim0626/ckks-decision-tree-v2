# CKKS Decision Tree Experiment Log

이 파일은 context compaction 이후에도 현재 연구/구현 상태를 이어가기 위한 작업 메모입니다.

## Current Project Goal

- 최종 목표: CKKS/FHE 기반 Decision Tree training을 fully encrypted 상태로 수행.
- 현재 단계: fully encrypted training으로 바로 가지 않고, 먼저 depth=1 encrypted MGII Decision Stump를 구현/검증 중.
- 중요한 원칙:
  - training core에서 plaintext `argmin`, decrypt, threshold 선택을 하지 않는다.
  - 학습된 model component도 가능한 ciphertext 상태로 유지한다.
  - debug decrypt는 client-side 검증 함수에서만 사용한다.

## Existing Inference Pipeline

기존 주요 파일:

- `ckks_tree.py`
  - DESILO FHE 기반 CKKS encrypted inference.
  - `predict_ckks`, `encrypted_argmax`, `sigmoid_approx_enc`.
  - GPU backend 사용: `Engine(..., mode="gpu", device_id=0)`.
- `train_tree.py`
  - sklearn Decision Tree training 후 `structure` 추출.
  - `soft_surrogate` threshold tuning 지원.
- `sigmoid_approx_coeffs.py`
  - sigmoid polynomial coefficient 관리.
  - 현재 5차 polynomial 사용.
- `main.py`
  - `iris`, `wine`, `breast_cancer` CKKS inference 실험 실행.

기존 `structure` 형식:

```python
{
    "children_left": ...,
    "children_right": ...,
    "feature": ...,
    "threshold": ...,
    "value": ...,
}
```

이 구조는 plaintext tree + encrypted input inference에는 맞지만, fully encrypted training 결과를 표현하기에는 부족하다. Fully encrypted training에서는 selected feature, threshold, leaf counts 등이 ciphertext로 유지되어야 한다.

## Current Sigmoid Setting

현재 `sigmoid_approx_coeffs.py`는 5차 sigmoid polynomial을 사용한다.

```python
SIGMOID_APPROX_DEGREE = 5
SIGMOID_COEFFS = [
    0.5,
    0.209637,
    0.0,
    -0.005402,
    0.0,
    0.000050,
]
```

15차 polynomial도 실험했었다.

- 15차는 `max_level=25`, `slot_count=32768`에서 bootstrapping 없이 실행 가능했다.
- 하지만 inference time이 크게 증가했다.
- 예: `breast_cancer` 기준 5차 약 `0.34s/sample`, 15차 약 `1.81s/sample`.
- 현재는 속도와 level budget을 고려해 5차로 되돌린 상태.

## GPU Status

DESILO FHE GPU backend 정상 사용 확인.

`nvidia-smi`에서 CKKS 실행 중:

```text
python process appears on RTX 4090
GPU-Util nonzero
```

현재 `ckks_tree.py`의 context:

```python
create_ckks_context(mode="gpu", device_id=0, max_level=15)
```

## Main Inference Experiments

`main.py`는 다음 dataset들을 순차 실행하도록 설정되어 있다.

```python
experiments = [
    ("iris", "minmax_minus1_1", "soft_surrogate"),
    ("wine", "minmax_minus1_1", "soft_surrogate"),
    ("breast_cancer", "minmax_minus1_1", "soft_surrogate"),
]
```

Hard DT 관련 출력은 제거했다.

제거된 출력:

- `Tuned hard DT accuracy`
- `train hard acc`
- threshold tuning node 상세 로그
- tree 구조 상세 출력

현재 CKKS inference는 hard Decision Tree가 아니라 soft Decision Tree이다.

```text
right_prob = sigmoid_poly(x - threshold)
left_prob = 1 - right_prob
```

따라서 CKKS accuracy가 Plain DT accuracy보다 높게 나오는 것도 가능하다. 이는 encryption이 accuracy를 올린 것이 아니라, soft traversal + tuned threshold가 hard split과 다른 classifier처럼 동작하기 때문이다.

## Plaintext MGII Trainer

한때 `mgii_train.py`를 만들어 plaintext MGII reference trainer를 구현했었다.

하지만 이후 목표가 다음처럼 정리되었다.

```text
plaintext에서 training 연산이 없어야 함
fully encrypted training 방향으로 가야 함
```

그래서 `mgii_train.py`는 삭제했다.

현재 repository에는 `mgii_train.py`가 없어야 한다.

## Fully Encrypted MGII Direction

Fully encrypted training은 기존 recursive tree training 방식과 다르다.

Plaintext recursion이 어려운 이유:

- plaintext `if split improves` 불가
- plaintext `argmin` 불가
- plaintext threshold 선택 불가
- learned model structure 노출 문제

따라서 방향:

```text
fixed depth full tree
public candidate grid
encrypted MGII score
encrypted argmin selector
encrypted selected threshold
encrypted feature selector
encrypted leaf counts
encrypted model 유지
```

목표는 (A) 완전 비공개 방식:

```text
candidate index 복호화 금지
selected threshold 복호화 금지
selected feature 복호화 금지
model을 ciphertext 상태로 유지
```

## Current New File: encrypted_mgii_stump.py

현재 새 파일:

```text
encrypted_mgii_stump.py
```

목적:

- depth=1 encrypted MGII Decision Stump prototype.
- training core에서 plaintext `argmin`/decrypt를 사용하지 않는다.
- client-side debug decrypt는 검증 목적으로만 분리되어 있다.

현재 구현된 dataclass:

```python
PublicSplitCandidate
EncryptedTrainingContext
EncryptedDataset
EncryptedSplitEvaluation
EncryptedStumpModel
```

현재 구현된 핵심 함수:

```python
create_encrypted_training_context()
encrypt_training_data()
make_public_grid_candidates()
make_small_public_threshold_grid()
encrypted_slot_sum()
encrypted_class_counts()
encrypted_mgii_from_counts()
evaluate_encrypted_candidate()
encrypted_argmin_selectors()
encrypted_select_threshold()
encrypted_feature_selectors()
encrypted_selected_feature_value()
train_encrypted_mgii_stump()
encrypt_sample()
encrypted_stump_predict_scores()
debug_decrypt_stump()
```

## Stump Smoke Test

실행:

```bash
python encrypted_mgii_stump.py
```

정상 출력 예:

```text
[encrypted MGII stump smoke test]
mode=gpu | candidates=4 | samples=4
training core completed without plaintext argmin/decrypt
encrypted selectors: 4
encrypted selected threshold: ready
encrypted feature selectors: 2
encrypted left leaf counts: 2
encrypted right leaf counts: 2
encrypted prediction scores: 2
```

## Debug Decrypt Result

Debug decrypt는 training core가 아니라 client-side 검증용이다.

확인한 내용:

- plaintext MGII score
- encrypted MGII score decrypt
- encrypted selector decrypt
- selected threshold decrypt
- feature selector decrypt
- encrypted prediction score decrypt

결과:

```text
plain_mgii와 enc_mgii는 거의 일치
```

예:

```text
candidate 0:
plain_mgii=1.480306
enc_mgii=1.480308
```

즉 encrypted MGII score 계산은 정상이다.

## Current Issue: Argmin Selector Softness

현재 pairwise-product argmin selector 방식:

```text
selector_i = product_j sigmoid(score_j - score_i)
```

문제:

- best candidate에 가장 큰 selector가 나오긴 한다.
- 하지만 selector가 충분히 one-hot하지 않다.
- 두 번째 후보가 많이 섞인다.

예:

```text
candidate 0 selector = 0.733738
candidate 1 selector = 0.333369
candidate 2 selector = 0.000623
candidate 3 selector = 0.000623
```

이 때문에 selected threshold가 정확한 candidate threshold가 아니라 섞여 나온다.

```text
expected threshold = -0.333333
selected threshold decrypt = -0.133456
```

결론:

```text
encrypted MGII score는 맞음
best candidate 방향도 맞음
하지만 pairwise-product selector가 one-hot이 아님
```

이 상태로 depth=2, depth=3을 진행하면 feature/threshold/weight propagation 오차가 누적된다.

## Tournament Argmin Progress

pairwise-product argmin 대신 encrypted tournament argmin을 구현했다.

추가된 핵심 함수:

```python
EncryptedTournamentWinner
encrypted_candidate_winner_state()
encrypted_blend()
encrypted_tournament_compare()
encrypted_tournament_argmin()
```

현재 `train_encrypted_mgii_stump()`의 기본 selection method:

```python
selection_method="tournament"
argmin_compare_scale=30.0
```

Tournament 방식:

```text
candidate를 2개씩 비교
더 작은 encrypted MGII score를 가진 쪽을 encrypted blend로 선택
threshold / feature selector / left counts / right counts를 함께 blend
winner index는 복호화하지 않음
```

현재 smoke test 결과:

```text
selection_method=tournament
pairwise selectors stored: 0
plain_mgii와 enc_mgii는 여전히 거의 일치
expected plaintext best index: 0
selected threshold decrypt: 약 -0.182
feature selector decrypt: [0.96269, 0.03731]
```

해석:

- encrypted MGII score 계산은 정상.
- tournament가 best feature 쪽으로 더 명확히 몰리긴 한다.
- 하지만 selected threshold는 아직 정확한 candidate threshold `-0.333333`까지 가지 못하고 soft하게 섞인다.
- `compare_scale=40` 이상은 sigmoid polynomial 근사 구간을 벗어나 값이 폭주했다.
- 현재 5차 sigmoid 기준으로는 `compare_scale=30` 근처가 synthetic smoke test에서 비교적 안정적이다.

남은 문제:

- candidate 수 n에 대해 O(n^2) 비교.
- selector product가 길어짐.
- candidate 수가 늘면 depth/time 폭발.
- selector가 soft하게 섞임.

위 O(n^2) 문제는 tournament로 완화됐지만, selector softness는 완전히 해결되지 않았다.

다음 후보:

```text
1. candidate grid를 더 작고 score 차이가 크게 나도록 설계
2. argmin comparison 전 MGII score scaling/normalization 개선
3. argmin comparison에만 더 높은 degree sigmoid 사용 검토
4. depth=1에서 iris subset으로 확장하기 전 synthetic case를 더 다양하게 테스트
```

Tournament output은 plaintext index가 아니라 다음 ciphertext들이다.

```text
winner_score
winner_threshold
winner_feature_selectors
winner_left_counts
winner_right_counts
```

다음 작업:

1. synthetic cases를 늘려 tournament argmin stability 확인.
2. candidate 수를 4에서 6~8로 늘려 runtime/selector quality 확인.
3. 작은 iris subset에서 depth=1 encrypted stump 검증.
4. depth=2 fixed full tree로 확장.
