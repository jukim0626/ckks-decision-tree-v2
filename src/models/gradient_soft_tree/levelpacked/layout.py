"""levelpacked의 슬롯 레이아웃(순수 numpy, CKKS 무관 - CPU에서 바로 테스트 가능).

ROADMAP 2-1~2-3: 기존 packed/는 노드를 하나씩, threshold를 (노드 x feature)개 broadcast ciphertext로
처리했다. 여기서는 **한 tree level의 노드 x feature x 샘플을 가능한 한 적은 ciphertext에** 담고, 파라미터도 같은
레이아웃으로 저장한다.

**타일(feature 그룹 x 샘플 그룹)**: feature는 G개 그룹(그룹당 최대 fg = ceil(F/G)개, block 수 Fg = next_pow2(fg)),
샘플은 H개 그룹(그룹당 최대 B = next_pow2 슬롯)으로 나눈다. 데이터 ciphertext 하나 = (feature 그룹 g, 샘플 그룹 h).
(G, H)는 데이터셋 크기(F, n, K, depth)만 보고 **모든 level의 ciphertext 수가 최소**가 되도록 자동 선택한다
(`_plan`) - 작은 데이터셋은 (1, 1)이라 타일 개념이 없는 것과 같다. 크기 정보(F, n)는 공개라 client가 암호화 전에
이 배치대로 평문을 놓으면 되므로 나눠 담는 비용은 없다.

**segment 레이아웃**: P = Fg * B. 노드 m(ciphertext 안 위치)의 segment는 슬롯 [m*P, (m+1)*P), 그 안에서 그룹의
로컬 feature j의 block은 [m*P + j*B, +B), 샘플 s는 그 block의 s번째 슬롯. ciphertext 하나에 c = slot_count // P개
노드가 들어간다(넘치면 c개씩 chunk).
- 데이터 X[g][h]: (m, j, s) = x[h*B + s, g의 j번째 feature] (노드마다 같은 데이터 복제)
- threshold/alpha[g]: (m, j, s) = 값을 block 전체(s=0..B-1)에 복제 - 샘플 그룹 h와 무관하게 공유
- leaf logits/class 분포: feature 대신 class c를 block 축으로 쓴다(Kp = next_pow2(K) <= Fg 필요)
- 노드별 샘플 벡터(gate, reach prob, g)[h]: (m, j=0, s)에만 값 ("node-sample" 레이아웃, feature 그룹과 무관)
- 예측 점수 y_hat/라벨 Y[h]: segment 0의 (c, s)
feature 축 합(gate, softmax 분모)은 그룹별 합을 ciphertext끼리 더하고, 샘플 축 합(gradient)은 샘플 그룹별 값을
더한 뒤 한 번 reduction 한다 - 결과가 같은 위치에 놓이므로 회전 없는 덧셈이다.

**노드 순서**: level l의 노드 m의 왼쪽 자식은 level l+1의 m, 오른쪽 자식은 m + 2^l -> level 전체의 자식 배치가
회전 한 번. 보통의 heap 순서(자식 2h, 2h+1)와는 비트가 뒤집힌 관계(m = bitreverse(h))라 평문 비교 때만 변환한다.

**파라미터 저장소(store)**: threshold/alpha(level x chunk x feature 그룹)와 leaf logits(chunk)를 "item"으로 보고,
first-fit decreasing으로 최소 개수의 ciphertext에 segment 단위로 채운다. 업데이트/momentum/clamp는 store
ciphertext 단위라 store 수 = iteration당 clamp 횟수다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from math import ceil

import numpy as np


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def bitreverse(x: int, bits: int) -> int:
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


@dataclass(frozen=True)
class Item:
    kind: str     # "threshold" | "alpha" | "leaf"
    level: int    # internal level(0..depth-1), leaf는 depth
    chunk: int    # level 안에서 몇 번째 chunk인지
    group: int    # feature 그룹(leaf는 0)
    count: int    # 이 chunk의 노드(leaf) 수
    store: int = -1   # 저장소 ciphertext 번호
    offset: int = -1  # 그 ciphertext 안의 시작 segment


@dataclass(frozen=True)
class Layout:
    n_samples: int
    n_features: int
    n_classes: int
    depth: int
    slot_count: int
    block: int = 0            # 0이면 자동. 추론 때는 학습과 같은 값을 지정한다.
    n_feature_groups: int = 0  # 0이면 자동. 추론 때는 학습과 같은 값을 지정한다.
    n_models: int = 1          # multi-start: 초기값만 다른 트리 R개를 같은 ciphertext에서 동시에 학습(2의 거듭제곱, R <= c)
    items: tuple = field(default=(), compare=False)

    # ---- (feature 그룹, 샘플 그룹) 선택 ----
    @cached_property
    def _plan(self) -> tuple[int, int, int]:
        """반환 (B, G, fg). block/n_feature_groups가 지정되면 그대로, 아니면 ciphertext 수 최소화."""
        Kp = next_pow2(self.n_classes)
        if self.block and self.n_feature_groups:
            return self.block, self.n_feature_groups, ceil(self.n_features / self.n_feature_groups)
        best = None
        B = 1
        while B <= self.slot_count:
            H = ceil(self.n_samples / B)
            for G in range(1, self.n_features + 1):
                fg = ceil(self.n_features / G)
                if (G - 1) * fg >= self.n_features:   # 빈 그룹이 생기는 G는 제외
                    continue
                Fg = next_pow2(fg)
                if Fg * B > self.slot_count or Kp > Fg:
                    continue
                c = self.slot_count // (Fg * B)
                if c < self.n_models:
                    continue
                R = self.n_models
                n_ct = H * G * sum(ceil((R << l) / c) for l in range(self.depth)) + H * ceil((R << self.depth) / c)
                key = (n_ct, G * H, G)
                if best is None or key < best[0]:
                    best = (key, (B, G, fg))
            B *= 2
        if best is None:
            raise ValueError("어떤 (feature 그룹, 샘플 그룹) 조합으로도 슬롯에 안 들어간다")
        return best[1]

    @cached_property
    def B(self) -> int:
        return self._plan[0]

    @cached_property
    def G(self) -> int:
        return self._plan[1]

    @cached_property
    def fg(self) -> int:
        return self._plan[2]

    @cached_property
    def H(self) -> int:
        return ceil(self.n_samples / self.B)

    @cached_property
    def Fg(self) -> int:
        return next_pow2(self.fg)

    @cached_property
    def Kp(self) -> int:
        return next_pow2(self.n_classes)

    @cached_property
    def P(self) -> int:
        return self.Fg * self.B

    @cached_property
    def c(self) -> int:
        """ciphertext 하나에 들어가는 노드(segment) 수."""
        return self.slot_count // self.P

    def validate(self) -> None:
        if self.c < 1:
            raise ValueError(f"노드 하나(segment P={self.P})가 slot_count={self.slot_count}보다 크다")
        if self.Kp > self.Fg:
            raise ValueError(f"class block 수(Kp={self.Kp})가 feature block 수(Fg={self.Fg})보다 많다")
        R = self.n_models
        if R < 1 or R & (R - 1) or R > self.c:
            raise ValueError(f"n_models({R})는 2의 거듭제곱이고 ciphertext당 노드 수 c({self.c}) 이하여야 한다")

    def features_of(self, g: int) -> range:
        return range(g * self.fg, min((g + 1) * self.fg, self.n_features))

    def samples_of(self, h: int) -> range:
        return range(h * self.B, min((h + 1) * self.B, self.n_samples))

    # ---- level/chunk 구조 ----
    def level_size(self, level: int) -> int:
        """level의 노드 수(모델 R개 전체). 노드 m은 모델 m % R에 속한다(root가 R개인 숲으로 보면 자식 배치 규칙이 그대로 성립)."""
        return self.n_models << level

    def chunk_count(self, level: int) -> int:
        return min(self.level_size(level), self.c)

    def n_chunks(self, level: int) -> int:
        return self.level_size(level) // self.chunk_count(level)

    @cached_property
    def x_segments(self) -> int:
        """데이터 X를 몇 노드분 복제해 둘지 - internal level chunk 중 가장 큰 것."""
        return self.chunk_count(self.depth - 1)

    # ---- 저장소 배치 ----
    def with_store(self) -> "Layout":
        raw = []
        for level in range(self.depth):
            for kind in ("threshold", "alpha"):
                for k in range(self.n_chunks(level)):
                    for g in range(self.G):
                        raw.append(Item(kind, level, k, g, self.chunk_count(level)))
        for k in range(self.n_chunks(self.depth)):
            raw.append(Item("leaf", self.depth, k, 0, self.chunk_count(self.depth)))
        order = sorted(range(len(raw)), key=lambda i: -raw[i].count)
        used: list[int] = []
        placed = [None] * len(raw)
        for i in order:
            it = raw[i]
            for q, u in enumerate(used):
                if u + it.count <= self.c:
                    placed[i] = Item(it.kind, it.level, it.chunk, it.group, it.count, q, u)
                    used[q] += it.count
                    break
            else:
                placed[i] = Item(it.kind, it.level, it.chunk, it.group, it.count, len(used), 0)
                used.append(it.count)
        return Layout(self.n_samples, self.n_features, self.n_classes, self.depth, self.slot_count,
                      self.B, self.G, self.n_models, tuple(placed))

    @cached_property
    def n_store(self) -> int:
        return 1 + max(it.store for it in self.items)

    @cached_property
    def _item_index(self) -> dict:
        return {(it.kind, it.level, it.chunk, it.group): it for it in self.items}

    def item(self, kind: str, level: int, chunk: int, group: int = 0) -> Item:
        return self._item_index[(kind, level, chunk, group)]

    # ---- 평문 마스크 ----
    def _mask(self, count: int, blocks: int, samples: int) -> np.ndarray:
        """segment m<count, block j<blocks, slot s<samples 인 곳만 1."""
        m = np.zeros(self.slot_count)
        for seg in range(count):
            for j in range(blocks):
                base = seg * self.P + j * self.B
                m[base: base + samples] = 1.0
        return m

    def feat_mask(self, count: int, g: int, h: int) -> np.ndarray:   # 데이터가 있는 곳
        return self._mask(count, len(self.features_of(g)), len(self.samples_of(h)))

    def feat_block_mask(self, count: int, g: int) -> np.ndarray:     # 파라미터 복제 영역
        return self._mask(count, len(self.features_of(g)), self.B)

    def class_block_mask(self, count: int) -> np.ndarray:
        return self._mask(count, self.n_classes, self.B)

    def node0_mask(self, count: int, h: int) -> np.ndarray:          # node-sample 레이아웃
        return self._mask(count, 1, len(self.samples_of(h)))

    def block0_mask(self, count: int) -> np.ndarray:                 # j=0 block 전체
        return self._mask(count, 1, self.B)

    def block_start_mask(self) -> np.ndarray:
        m = np.zeros(self.slot_count)
        m[:: self.B] = 1.0
        return m

    def y_mask(self, h: int) -> np.ndarray:                          # y_hat/Y: segment r(모델), c<K, s<n_h
        return self._mask(self.n_models, self.n_classes, len(self.samples_of(h)))

    def item_mask(self, it: Item) -> np.ndarray:
        return self.class_block_mask(it.count) if it.kind == "leaf" else self.feat_block_mask(it.count, it.group)

    def store_vector(self, per_kind: dict) -> list[np.ndarray]:
        """저장소 ciphertext마다, 각 item 영역에 per_kind[kind] 값을 채운 평문 벡터(lr, clamp 경계 등)."""
        out = [np.zeros(self.slot_count) for _ in range(self.n_store)]
        for it in self.items:
            region = np.roll(self.item_mask(it), it.offset * self.P)
            out[it.store] = np.where(region > 0, per_kind[it.kind], out[it.store])
        return out

    # ---- 압축 저장(compact): clamp/momentum을 ciphertext 1개에서 하기 위함 ----
    # store ciphertext 하나에는 값이 block마다 B번 복제돼 있어 실제 값은 M = slot_count // B개뿐이다.
    # 각 block b의 값을 슬롯 b로 모으면(압축) store q는 compact ciphertext의 [q*M, (q+1)*M)를 차지한다.
    # 이동은 b의 비트별로 log2(M)단계: k번째 비트가 1인 원소만 -2^k*(B-1) 회전(마스크 1레벨/단계).
    # 단계 k 직전 원소 b의 위치 = b*B - (b mod 2^k)*(B-1) (서로 겹치지 않음).
    @cached_property
    def M(self) -> int:
        return self.slot_count // self.B

    @cached_property
    def compact_bits(self) -> int:
        return max(self.M - 1, 0).bit_length()

    @cached_property
    def use_compact(self) -> bool:
        """store가 2개 이상일 때만 압축이 이득(1개면 압축/확장 비용만 추가)."""
        return self.n_store > 1

    @cached_property
    def n_compact(self) -> int:
        return ceil(self.n_store * self.M / self.slot_count)

    @cached_property
    def n_params(self) -> int:
        """저장되는 파라미터 ciphertext 수(압축 모드면 compact 수)."""
        return self.n_compact if self.use_compact else self.n_store

    def compact_pos(self, q: int) -> tuple[int, int]:
        """store q가 들어가는 (compact ciphertext 번호, 시작 슬롯)."""
        start = q * self.M
        return start // self.slot_count, start % self.slot_count

    def _pos_before(self, b: int, k: int) -> int:
        return b * self.B - (b % (1 << k)) * (self.B - 1)

    def compact_step_masks(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """단계 k(압축 방향) 직전 위치 기준: (비트 k가 1인 원소 마스크, 0인 원소 마스크)."""
        on, off = np.zeros(self.slot_count), np.zeros(self.slot_count)
        for b in range(self.M):
            (on if (b >> k) & 1 else off)[self._pos_before(b, k)] = 1.0
        return on, off

    def expand_step_masks(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """단계 k(확장 방향, 높은 비트부터) 직전 위치 = 압축 단계 k 직후 위치."""
        on, off = np.zeros(self.slot_count), np.zeros(self.slot_count)
        for b in range(self.M):
            (on if (b >> k) & 1 else off)[self._pos_before(b, k + 1)] = 1.0
        return on, off

    def compact_vector(self, store_vectors: list[np.ndarray]) -> list[np.ndarray]:
        """store 모양 평문 벡터들(lr, clamp 경계 등) -> compact 모양."""
        out = [np.zeros(self.slot_count) for _ in range(self.n_compact)]
        for q, v in enumerate(store_vectors):
            ci, start = self.compact_pos(q)
            out[ci][start: start + self.M] = v[:: self.B][: self.M]
        return out

    def pack_params(self, params: dict) -> list[np.ndarray]:
        """client 측 초기 파라미터 배치: 압축 모드면 compact, 아니면 store 모양."""
        st = self.pack_store(params)
        return self.compact_vector(st) if self.use_compact else st

    def unpack_params(self, vecs: list[np.ndarray]) -> dict:
        if not self.use_compact:
            return self.unpack_store(vecs)
        store = []
        for q in range(self.n_store):
            ci, start = self.compact_pos(q)
            v = np.zeros(self.slot_count)
            v[:: self.B] = vecs[ci][start: start + self.M]
            store.append(v)
        return self.unpack_store(store)

    # ---- 평문 <-> 슬롯 변환(client 측) ----
    def heap_index(self, level: int, m: int) -> int:
        """levelpacked 노드 번호 m -> reference(heap 순서) 노드 번호."""
        return (1 << level) - 1 + bitreverse(m, level)

    def pack_X(self, X: np.ndarray, g: int, h: int) -> np.ndarray:
        v = np.zeros(self.slot_count)
        rows = X[self.samples_of(h).start: self.samples_of(h).stop]
        for seg in range(self.x_segments):
            for j, f in enumerate(self.features_of(g)):
                base = seg * self.P + j * self.B
                v[base: base + len(rows)] = rows[:, f]
        return v

    def pack_Y(self, Y: np.ndarray, h: int) -> np.ndarray:
        """모델마다 같은 라벨을 segment r에 복제."""
        v = np.zeros(self.slot_count)
        rows = Y[self.samples_of(h).start: self.samples_of(h).stop]
        for r in range(self.n_models):
            for c in range(self.n_classes):
                base = r * self.P + c * self.B
                v[base: base + len(rows)] = rows[:, c]
        return v

    def _item_slots(self, it: Item):
        """item 안의 (모델 r, heap 노드 번호, 열 번호(feature 또는 class), 시작 슬롯) 나열."""
        R = self.n_models
        for local in range(it.count):
            m = it.chunk * self.chunk_count(it.level) + local
            r, q = m % R, m // R
            seg = (it.offset + local) * self.P
            if it.kind == "leaf":
                h = bitreverse(q, self.depth)
                for c in range(self.n_classes):
                    yield r, h, c, seg + c * self.B
            else:
                h = self.heap_index(it.level, q)
                for j, f in enumerate(self.features_of(it.group)):
                    yield r, h, f, seg + j * self.B

    def pack_store(self, params) -> list[np.ndarray]:
        """reference 형식(heap 순서) params(모델 1개면 dict, R개면 dict 리스트) -> 저장소 벡터."""
        plist = params if isinstance(params, (list, tuple)) else [params]
        assert len(plist) == self.n_models, f"모델 {self.n_models}개 분의 파라미터가 필요"
        out = [np.zeros(self.slot_count) for _ in range(self.n_store)]
        for it in self.items:
            key = "leaf_logits" if it.kind == "leaf" else it.kind
            for r, h, col, start in self._item_slots(it):
                out[it.store][start: start + self.B] = plist[r][key][h, col]
        return out

    def unpack_store(self, store: list[np.ndarray]):
        """저장소 벡터 -> reference 형식(모델 1개면 dict, R개면 dict 리스트). 각 block의 첫 슬롯 값을 읽는다."""
        n_internal = (1 << self.depth) - 1
        outs = [{"alpha": np.zeros((n_internal, self.n_features)), "threshold": np.zeros((n_internal, self.n_features)),
                 "leaf_logits": np.zeros((1 << self.depth, self.n_classes))} for _ in range(self.n_models)]
        for it in self.items:
            key = "leaf_logits" if it.kind == "leaf" else it.kind
            for r, h, col, start in self._item_slots(it):
                outs[r][key][h, col] = store[it.store][start]
        return outs[0] if self.n_models == 1 else outs

    def describe(self) -> str:
        return (f"R={self.n_models} B={self.B} H={self.H} G={self.G}(fg={self.fg}, Fg={self.Fg}) P={self.P} c={self.c} "
                f"store={self.n_store if self.items else '?'}")

    def to_dict(self) -> dict:
        return {"n_samples": self.n_samples, "n_features": self.n_features, "n_classes": self.n_classes,
                "depth": self.depth, "slot_count": self.slot_count, "block": self.B, "n_feature_groups": self.G,
                "n_models": self.n_models}

    @staticmethod
    def from_dict(d: dict) -> "Layout":
        lay = Layout(d["n_samples"], d["n_features"], d["n_classes"], d["depth"], d["slot_count"],
                     d.get("block", 0), d.get("n_feature_groups", 0), d.get("n_models", 1))
        lay.validate()
        return lay.with_store()


def free_models(n_samples: int, n_features: int, n_classes: int, depth: int, slot_count: int = 32768, max_r: int = 8) -> int:
    """ciphertext 수(= 연산량)를 늘리지 않고 넣을 수 있는 최대 모델 수 R(2의 거듭제곱, max_r 이하).
    작은 데이터셋은 빈 슬롯이 많아 R이 커지고, 슬롯이 꽉 찬 데이터셋은 1."""
    def n_ct(R):
        lay = Layout(n_samples, n_features, n_classes, depth, slot_count, n_models=R)
        try:
            lay.validate()
        except ValueError:
            return None
        return lay.H * lay.G * sum(ceil((R << l) / lay.c) for l in range(depth)) + lay.H * ceil((R << depth) / lay.c)
    base, best, R = n_ct(1), 1, 2
    while R <= max_r:
        v = n_ct(R)
        if v is None or v > base:
            break
        best, R = R, R * 2
    return best
