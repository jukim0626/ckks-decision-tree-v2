# Agent Memory

이 파일은 이 repository에서 작업하는 AI coding agent가 사용자 선호와 환경을 빠르게 파악하기 위한 메모입니다.

## About Me

- 학부연구생
- 동형암호 연구실 소속
- 지도교수: 이용우
- 코딩 초보
- 사용 장비: M1 MacBook Air

## Communication

- 답변은 한국어로 작성합니다.
- 수학, 암호, ML 용어는 영어 표현을 유지합니다.
  - 예: FHE, CKKS, ciphertext, plaintext, Decision Tree, gradient, inference
- 코드 설명은 학습 목적에 맞게 line by line으로 자세히 설명합니다.

## Action Descriptions

- Bash 명령어, 파일 편집, fetch 등 모든 도구 사용 시 description은 한국어로 설명합니다.
- 명령어 자체는 영어 그대로 유지합니다.
- 예:
  - "venv에 패키지가 설치됐는지 확인"
  - "현재 repository의 Python 파일 목록 확인"
  - "테스트 스크립트 실행"

## Code Style

- Python 3.10+ 기준으로 작성합니다.
- type hints를 사용합니다.
- 함수와 클래스에는 docstring을 작성합니다.
- 변수명은 영어 `snake_case`를 사용합니다.
- 주석은 짧은 한국어로 작성해도 됩니다.

## Working Style

- 큰 변경 전에는 plan을 먼저 제시합니다.
- 새로운 개념이나 라이브러리를 사용할 때는 짧은 배경 설명을 먼저 제공합니다.
- 변경 작업은 step-by-step으로 진행합니다.
- 사용자가 학습할 수 있도록 왜 그렇게 수정하는지 함께 설명합니다.

## Environment

- OS: macOS Sequoia
- CPU/Architecture: Apple Silicon M1
- Shell: zsh
- Terminal: Warp
- Editor: VSCode
- Python: Homebrew Python
- Python environment: project별 venv 사용
- M1 MacBook Air에는 NVIDIA GPU가 없으므로 heavy workload는 연구실 서버 사용을 전제로 합니다.

## Research Context

- 주요 연구 관심사: FHE 기반 privacy-preserving ML
- 현재 관심 scheme: CKKS scheme
- 현재 전환 작업: TenSEAL 기반 구현에서 DESILO FHE 기반 구현으로 전환 중
- 관련 키워드:
  - FHE
  - CKKS
  - privacy-preserving ML
  - encrypted inference
  - ciphertext computation
  - Decision Tree
