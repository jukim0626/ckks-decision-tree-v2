"""closed-form MGI tree의 학습 결과를 담는 모델. client_assisted/models.py의
EncryptedFixedDepthTreeModel(depth, node_splits, leaf_counts) 패턴을 그대로 따르되,
이 방식은 "candidate 하나를 선택"하지 않고 항상 모든 feature의 gate를 blend하므로
node_splits(선택된 split 하나) 대신 노드별 (thresholds, soft-MGI weights)를 저장한다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClosedFormMgiNode:
    """한 노드의 라우팅 정보. 새 dataset(train/test/임의 샘플)의 gate를 재계산하는 데
    필요한 것만 담는다 - thresholds로 feature별 gate를 만들고, weights로 그 gate들을
    blend한다 (둘 다 학습에서 재사용, 새로 학습하지 않음).

    2026-08-11: ciphertext 객체를 직접 들고 있지 않고 **파일 경로**만 들고 있다 - key가
    살아있는 프로세스가 ciphertext를 한 번이라도 읽으면(fresh read든 to_cuda든) 그 메모리
    풀이 계속 붙잡혀서, 학습 중 부모 프로세스가 노드마다 결과를 즉시 read_ciphertext()로
    읽어들이면 부모 자체가 다시 누적 OOM에 걸렸다(node_worker.py로 노드 "계산"을 격리한
    것만으로는 부족했음). 그래서 inference.py가 실제로 그 노드를 쓸 때만
    ctx.engine.read_ciphertext(path)로 그때그때 읽고 바로 버리도록 미룬다."""

    thresholds: list  # feature별 threshold ciphertext 파일 경로(Path), pre-order
    weights: object  # soft_mgi_weights() 결과 ciphertext 파일 경로(Path)


@dataclass
class ClosedFormMgiTreeModel:
    depth: int
    nodes: list  # ClosedFormMgiNode, pre-order (내부 노드만, 길이 (1<<depth)-1)
    leaf_counts: list  # 리프별 [class별 ciphertext 파일 경로(Path)], pre-order 리프 순서
    session_dir: object  # nodes/leaf_counts가 가리키는 ciphertext 파일들이 있는 디렉터리 -
    # 모델을 다 쓴 뒤 호출자가 shutil.rmtree(model.session_dir)로 정리해야 한다.
