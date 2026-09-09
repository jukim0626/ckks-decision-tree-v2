"""results 표의 Mult/Rotation 열을 채우기 위한 얇은 call-counting proxy.

closed_form_mgi 세션에서 한때 쓰였던 'CountingEngineProxy'(EXPERIMENT_LOG.md 2026-08-12
"오진" 항목 - 그 사건과 무관하게 이번엔 단순 카운팅 용도로만 씀, GPU 메모리 문제와는
관계없다)와 같은 아이디어를 최소 형태로 재구현했다. __getattr__로 실제 engine에 위임하되
multiply/rotate/sum 호출만 가로채 카운트한다."""

from __future__ import annotations


class CountingEngineProxy:
    def __init__(self, engine):
        self._engine = engine
        self.counts = {"multiply": 0, "rotate": 0, "sum": 0, "bootstrap": 0, "evaluate_polynomial": 0}

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def multiply(self, *args, **kwargs):
        self.counts["multiply"] += 1
        return self._engine.multiply(*args, **kwargs)

    def rotate(self, *args, **kwargs):
        self.counts["rotate"] += 1
        return self._engine.rotate(*args, **kwargs)

    def sum(self, *args, **kwargs):
        # engine.sum()은 내부적으로 log2(slot_count)번 rotate+add를 한다 - rotation 비용의
        # 실제 대부분이 여기서 나온다(baseline profile의 "sum(rotation 기반 합산) 207회"와
        # 대응). 여기서는 "호출 횟수"만 세고, rotation "연산 횟수"는 별도로 근사하지 않는다
        # (engine 내부 구현에 접근할 수 없어 정확한 log2(slot_count) 곱을 재현하려면 값이
        # 임의로 보일 수 있어 - count만 정직하게 보고한다).
        self.counts["sum"] += 1
        return self._engine.sum(*args, **kwargs)

    def bootstrap(self, *args, **kwargs):
        self.counts["bootstrap"] += 1
        return self._engine.bootstrap(*args, **kwargs)

    def evaluate_polynomial(self, *args, **kwargs):
        self.counts["evaluate_polynomial"] += 1
        return self._engine.evaluate_polynomial(*args, **kwargs)
