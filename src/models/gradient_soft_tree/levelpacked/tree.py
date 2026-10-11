"""levelpacked soft decision tree의 forward/backward/update (CKKS 또는 numpy 시뮬레이터).

수식은 `baseline/reference.py`의 `compute_gradients`(poly 모드)와 같다 - 다른 건 데이터 배치뿐이다
(`layout.py` 참고). level 하나를 ciphertext 몇 개로 한 번에 처리하므로 step당 다항식 평가/Newton/bootstrap
횟수가 노드 수(2^depth)가 아니라 level 수(depth)에 비례한다(chunk가 여러 개인 넓은 데이터셋은 chunk 수배).

파라미터는 `Layout.items`가 정한 저장소 ciphertext(store)에 들어있다. 매 iteration:
  1) store에서 level별 threshold/alpha chunk, leaf chunk를 꺼낸다(회전 + 마스크)
  2) forward(level 단위) -> y_hat -> MSE gradient -> backward(level 단위, 역순)
  3) gradient를 store와 같은 모양으로 모은다(회전 + 덧셈)
  4) update: GD 또는 momentum, 선택적으로 합성 다항식 clamp(구간 안전, ROADMAP B)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.gradient_soft_tree.levelpacked import he_ops as H
from models.gradient_soft_tree.levelpacked.layout import Layout


@dataclass(frozen=True)
class ApproxSpec:
    sigmoid_coeffs: tuple
    steepness: float
    exp_coeffs: tuple
    exp_interval: float
    recip_iterations: int


@dataclass(frozen=True)
class OptConfig:
    optimizer: str = "gd"        # gd | mom
    lr: float = 2.0
    beta: float = 0.9
    kappa: float = 0.0           # 노드(threshold/alpha) lr = lr * max(1, kappa * n_features)
    clamp: bool = False          # 매 iteration 합성 다항식 clamp로 구간 보장
    threshold_bound: float = 0.95   # |t| <= 이 값 (x in [-1,1]이면 |x-t| <= 1.95 < 2)
    threshold_radius: float = 8.0
    softmax_bound: float = 2.45     # |alpha|, |leaf_logit| <= 이 값
    softmax_radius: float = 14.0
    leaf_k_scale: bool = False   # leaf lr = lr * n_classes (leaf 분포가 K개로 퍼져 gradient가 ~1/K로 작아지는 것 보정, 10-08)
    balance: float = 0.0         # 나눗셈 없는 balance penalty λ (10-09, reference.compute_gradients의 balance_lambda와 같은 식)

    def node_lr(self, n_features: int) -> float:
        return self.lr * max(1.0, self.kappa * n_features)

    def leaf_lr(self, n_classes: int) -> float:
        return self.lr * (n_classes if self.leaf_k_scale else 1.0)


def default_spec() -> ApproxSpec:
    from core.approximation.sigmoid import SPLIT_SIGMOID_COEFFS, STEEPNESS
    from core.encrypted_ops.softmax import _EXP_COEFFS, SOFTMAX_EXP_INTERVAL, SOFTMAX_RECIP_ITERATIONS
    return ApproxSpec(tuple(SPLIT_SIGMOID_COEFFS), STEEPNESS, tuple(_EXP_COEFFS), SOFTMAX_EXP_INTERVAL[1], SOFTMAX_RECIP_ITERATIONS)


# ---------------------------------------------------------------- store <-> item
def extract(ctx, lay: Layout, store: list, kind: str, level: int, chunk: int, group: int = 0):
    it = lay.item(kind, level, chunk, group)
    ct = H.rot(ctx, store[it.store], -it.offset * lay.P)
    return H.mul(ctx, ct, lay.item_mask(it))


def assemble(ctx, lay: Layout, parts: dict) -> list:
    """parts[(kind, level, chunk, group)] = item 모양 ct(유효 영역 밖 0) -> store 모양 ct 리스트."""
    out = [None] * lay.n_store
    for it in lay.items:
        ct = H.rot(ctx, parts[(it.kind, it.level, it.chunk, it.group)], it.offset * lay.P)
        out[it.store] = ct if out[it.store] is None else ctx.engine.add(out[it.store], ct)
    return out


def _sum(ctx, cts: list):
    out = cts[0]
    for ct in cts[1:]:
        out = ctx.engine.add(out, ct)
    return out


# ---------------------------------------------------------------- forward
# 표기: X[g][h] = feature 그룹 g x 샘플 그룹 h 데이터, probs[k][h] = chunk k의 샘플 그룹 h reach prob
def _forward(ctx, lay: Layout, spec: ApproxSpec, X, store, keep: bool):
    """반환: (leaf_prob[chunk][h], level별 캐시(backward용, keep=False면 None))."""
    eng = ctx.engine
    G_, H_ = range(lay.G), range(lay.H)
    cache = []
    probs = [[None] * lay.H]  # level 0: reach prob = 1 (암묵적)
    for level in range(lay.depth):
        count = lay.chunk_count(level)
        level_cache, lefts, rights = [], [], []
        for k in range(lay.n_chunks(level)):
            T = [extract(ctx, lay, store, "threshold", level, k, g) for g in G_]
            A = [extract(ctx, lay, store, "alpha", level, k, g) for g in G_]
            W = H.seg_softmax(ctx, A, lay, [len(lay.features_of(g)) for g in G_], count,
                              spec.exp_coeffs, spec.exp_interval, spec.recip_iterations)
            Gs, gates, left_k, right_k = [], [], [], []
            for h in H_:
                Gh = [H.poly(ctx, eng.subtract(X[g][h], T[g]), spec.sigmoid_coeffs, H.SIGMOID_MIN_LEVEL) for g in G_]
                # gate = sum_j w_j * sigmoid_j : 그룹 안 fold -> 그룹끼리 덧셈
                gate = _sum(ctx, [H.fold(ctx, H.mul(ctx, Gh[g], W[g]), lay.B, lay.Fg) for g in G_])
                nmask = lay.node0_mask(count, h)
                gate = H.mul(ctx, gate, nmask)
                p = probs[k][h]
                if p is None:
                    right, left = gate, eng.subtract(nmask, gate)
                else:
                    right = H.mul(ctx, p, gate)
                    left = eng.subtract(p, right)
                Gs.append(Gh)
                gates.append(gate)
                left_k.append(left)
                right_k.append(right)
            lefts.append(left_k)
            rights.append(right_k)
            if keep:
                level_cache.append({"W": W, "G": Gs, "gate": gates, "p": probs[k]})
        # 자식 배치: 왼쪽 자식 = 같은 자리, 오른쪽 자식 = +N 노드
        N = lay.level_size(level)
        if 2 * N <= lay.c:
            probs = [[eng.add(lefts[0][h], H.rot(ctx, rights[0][h], N * lay.P)) for h in H_]]
        else:
            probs = lefts + rights
        cache.append(level_cache)
    return probs, cache


def _leaf_scores(ctx, lay: Layout, spec: ApproxSpec, store, leaf_probs, keep: bool):
    """반환: (y_hat[h], leaf 캐시)."""
    count = lay.chunk_count(lay.depth)
    yhat, leaf_cache = [None] * lay.H, []
    for k, lp in enumerate(leaf_probs):
        L = extract(ctx, lay, store, "leaf", lay.depth, k)
        dist = H.seg_softmax(ctx, [L], lay, [lay.n_classes], count, spec.exp_coeffs, spec.exp_interval, spec.recip_iterations)[0]
        lpreps = []
        R = lay.n_models
        for h in range(lay.H):
            lprep = H.spread(ctx, lp[h], lay.B, lay.Kp)
            # 같은 모델의 leaf끼리만 합친다(R칸 간격) -> 모델 r의 y_hat이 segment r에 모임. R=1이면 기존과 같음.
            y_k = H.fold(ctx, H.mul(ctx, lprep, dist), R * lay.P, count // R)
            yhat[h] = y_k if yhat[h] is None else ctx.engine.add(yhat[h], y_k)
            lpreps.append(lprep)
        if keep:
            leaf_cache.append({"dist": dist, "lprep": lpreps})
    return yhat, leaf_cache


def predict_scores(ctx, lay: Layout, spec: ApproxSpec, X, store) -> list:
    """encrypted inference: 샘플 그룹별 클래스 점수 y_hat[h](segment 0의 (c, s)). 파라미터는 decrypt하지 않는다.
    store는 store 모양(압축 모드면 호출부가 materialize로 펼쳐서 넘긴다)."""
    leaf_probs, _ = _forward(ctx, lay, spec, X, store, keep=False)
    yhat, _ = _leaf_scores(ctx, lay, spec, store, leaf_probs, keep=False)
    if lay.n_models == 1:
        return [H.mul(ctx, yhat[h], lay.y_mask(h)) for h in range(lay.H)]
    # multi-start 앙상블: 모델 R개의 y_hat(segment 0..R-1)을 평균해서 segment 0으로 - 비교 연산 없음
    R = lay.n_models
    out = []
    for h in range(lay.H):
        y = H.mul(ctx, yhat[h], lay.y_mask(h))
        out.append(H.mul(ctx, H.fold(ctx, y, lay.P, R), lay._mask(1, lay.n_classes, len(lay.samples_of(h))) * (1.0 / R)))
    return out


# ---------------------------------------------------------------- forward + backward
def gradients(ctx, lay: Layout, spec: ApproxSpec, X, Y, store, balance: float = 0.0) -> list:
    """store와 같은 모양의 gradient ciphertext 리스트. X[g][h], Y[h].
    balance > 0이면 나눗셈 없는 balance penalty λ Σ_i 2^-l β_i², β_i = Σ_s P^i(s)(g_i(s) - 1/2)/n 의 gradient를
    dL/dgate_i(s)에 2λ 2^-l β_i P^i(s)/n로 더한다(P^i 경로는 stop-gradient - reference와 같은 근사)."""
    eng = ctx.engine
    G_, H_ = range(lay.G), range(lay.H)
    leaf_probs, cache = _forward(ctx, lay, spec, X, store, keep=True)
    yhat, leaf_cache = _leaf_scores(ctx, lay, spec, store, leaf_probs, keep=True)
    # dL/dy_hat (MSE, 전체 샘플 수로 나눔)
    E = [H.mul(ctx, eng.subtract(yhat[h], Y[h]), lay.y_mask(h) * (2.0 / lay.n_samples)) for h in H_]

    parts = {}
    # ---- leaf backward ----
    count = lay.chunk_count(lay.depth)
    g_next = []
    for k, lc in enumerate(leaf_cache):
        dist, lpreps = lc["dist"], lc["lprep"]
        ereps = [H.spread(ctx, E[h], lay.n_models * lay.P, count // lay.n_models) for h in H_]  # 모델 r의 E -> 모델 r의 leaf
        # dL/d dist_{l,c} = sum_s E_c(s) * leafprob_l(s): 샘플 그룹끼리 더한 뒤 한 번 reduction
        gdist = H.block_allreduce(ctx, _sum(ctx, [H.mul(ctx, ereps[h], lpreps[h]) for h in H_]), lay)
        dot = H.fold(ctx, H.mul(ctx, gdist, dist), lay.B, lay.Kp)
        dot = H.spread(ctx, H.mul(ctx, dot, lay.block0_mask(count)), lay.B, lay.Kp)
        parts[("leaf", lay.depth, k, 0)] = H.mul(ctx, dist, eng.subtract(gdist, dot))
        g_next.append([H.mul(ctx, H.fold(ctx, H.mul(ctx, ereps[h], dist), lay.B, lay.Kp), lay.node0_mask(count, h)) for h in H_])

    # ---- internal levels backward (깊은 level부터) ----
    for level in reversed(range(lay.depth)):
        count = lay.chunk_count(level)
        N = lay.level_size(level)
        if 2 * N <= lay.c:
            g_left = [[H.mul(ctx, g_next[0][h], lay.node0_mask(count, h)) for h in H_]]
            g_right = [[H.mul(ctx, H.rot(ctx, g_next[0][h], -N * lay.P), lay.node0_mask(count, h)) for h in H_]]
        else:
            half = len(g_next) // 2
            g_left, g_right = g_next[:half], g_next[half:]
        g_this = []
        for k, lc in enumerate(cache[level]):
            W, Gs, gates, ps = lc["W"], lc["G"], lc["gate"], lc["p"]
            dgreps, gate_reps, g_k = [], [], []
            if balance:
                # β_i: node-sample 레이아웃에서 샘플 합 -> block 안 모든 슬롯에 복제(샘플 그룹은 합하기 전에 더함)
                us = []
                for h in H_:
                    nm = lay.node0_mask(count, h)
                    centered = eng.subtract(gates[h], nm * 0.5)                      # g - 1/2 (유효 샘플만)
                    us.append(centered if ps[h] is None else H.mul(ctx, ps[h], centered))
                beta = H.block_allreduce(ctx, _sum(ctx, us), lay)
                bal_coef = 2.0 * balance * 2.0 ** (-level) / lay.n_samples / lay.n_samples   # β의 1/n과 항의 1/n
            for h in H_:
                diff = eng.subtract(g_right[k][h], g_left[k][h])
                dgate = diff if ps[h] is None else H.mul(ctx, ps[h], diff)
                if balance:
                    nm = lay.node0_mask(count, h)
                    pterm = H.mul(ctx, beta, nm * bal_coef) if ps[h] is None else H.mul(ctx, H.mul(ctx, beta, ps[h]), bal_coef)
                    dgate = eng.add(dgate, pterm)
                if level > 0:
                    g_k.append(eng.add(g_left[k][h], H.mul(ctx, gates[h], diff)))
                dgreps.append(H.spread(ctx, dgate, lay.B, lay.Fg))
                gate_reps.append(H.spread(ctx, gates[h], lay.B, lay.Fg))
            g_this.append(g_k)
            for g in G_:
                # 레벨 확보는 서로 독립인 ct끼리 묶어서 한다(bootstrap 2개씩 merge, 2026-10-09). 수식은 그대로:
                # threshold: -s * w_j * sum_s dgate * G(1-G),  alpha: w_j * sum_s dgate * (G_j - gate)
                t_terms, a_terms = [], []
                for h in H_:
                    dgreps[h], Gg, gate_reps[h] = H.lvs(ctx, [dgreps[h], Gs[h][g], gate_reps[h]], 7)
                    dg = dgreps[h]
                    t_terms.append(H.mul(ctx, H.mul(ctx, dg, H.mul(ctx, Gg, eng.subtract(1.0, Gg))),
                                         lay.feat_mask(count, g, h) * (-spec.steepness)))
                    a_terms.append(H.mul(ctx, H.mul(ctx, dg, eng.subtract(Gg, gate_reps[h])), lay.feat_mask(count, g, h)))
                t_sum, a_sum = H.lvs(ctx, [_sum(ctx, t_terms), _sum(ctx, a_terms)], 5)
                t_red, a_red = H.block_allreduce(ctx, t_sum, lay), H.block_allreduce(ctx, a_sum, lay)
                Wg, t_red, a_red = H.lvs(ctx, [W[g], t_red, a_red])
                parts[("threshold", level, k, g)] = H.mul(ctx, Wg, t_red)
                parts[("alpha", level, k, g)] = H.mul(ctx, Wg, a_red)
        g_next = g_this
    return assemble(ctx, lay, parts)


# ---------------------------------------------------------------- 압축/확장
# M(= store ct 하나의 block 수)이 작으면 "회전 M번 + 레벨 1"이 "log2(M)단계, 단계마다 레벨 1"보다 싸다(회전은 bootstrap보다
# 훨씬 싸고, 저장소 ct가 많은 넓은 데이터셋일수록 B가 커서 M이 작다). 2026-10-08 soybean(M=32) 실측에서 log 방식은
# 압축/확장에 bootstrap이 store ct마다 들어 기대만큼 빠르지 않았다.
DIRECT_MAX_M = 64


def _block_moves(lay: Layout):
    """(b, 압축 방향 이동량): block b의 값을 슬롯 b*B -> b로."""
    return [(b, -b * (lay.B - 1)) for b in range(lay.M)]


def compact(ctx, lay: Layout, store: list) -> list:
    """store 모양 ct들 -> compact ct들(block마다 값 1개, 연속 슬롯)."""
    out = [None] * lay.n_compact
    direct = lay.M <= DIRECT_MAX_M
    for q, ct in enumerate(store):
        if direct:  # 레벨 1: sum_b rot(mask_b * x, -b(B-1))
            ct = H.lv(ctx, ct)  # 레벨 확보는 루프 밖에서 한 번만(안에서 하면 block마다 bootstrap)
            x = None
            for b, shift in _block_moves(lay):
                m = np.zeros(lay.slot_count)
                m[b * lay.B] = 1.0
                piece = H.rot(ctx, ctx.engine.multiply(ct, m), shift)
                x = piece if x is None else ctx.engine.add(x, piece)
        else:       # 레벨 1 + log2(M)
            x = H.mul(ctx, ct, lay.block_start_mask())
            for k in range(lay.compact_bits):
                on, off = lay.compact_step_masks(k)
                x = ctx.engine.add(H.mul(ctx, x, off), H.rot(ctx, H.mul(ctx, x, on), -(1 << k) * (lay.B - 1)))
        ci, start = lay.compact_pos(q)
        x = H.rot(ctx, x, start)
        out[ci] = x if out[ci] is None else ctx.engine.add(out[ci], x)
    return out


def expand(ctx, lay: Layout, params: list) -> list:
    """compact ct들 -> store 모양 ct들(값을 각 block 전체에 복제). 압축의 역순.
    펼치기 전에 compact ct를 한 번만 레벨 확보해 두면 store ct마다 따로 bootstrap하지 않는다."""
    direct = lay.M <= DIRECT_MAX_M
    need = 2 if direct else 2 + lay.compact_bits
    params = [H.lv(ctx, p, need + H.MIN_LEVEL) for p in params]
    store = []
    for q in range(lay.n_store):
        ci, start = lay.compact_pos(q)
        x = H.rot(ctx, params[ci], -start)
        if direct:  # 레벨 1: sum_b rot(mask_b * x, +b(B-1))
            y = None
            for b, shift in _block_moves(lay):
                m = np.zeros(lay.slot_count)
                m[b] = 1.0
                piece = H.rot(ctx, ctx.engine.multiply(x, m), -shift)
                y = piece if y is None else ctx.engine.add(y, piece)
            x = y
        else:
            region = np.zeros(lay.slot_count)
            region[: lay.M] = 1.0
            x = ctx.engine.multiply(x, region)
            for k in reversed(range(lay.compact_bits)):
                on, off = lay.expand_step_masks(k)
                x = ctx.engine.add(ctx.engine.multiply(x, off), H.rot(ctx, ctx.engine.multiply(x, on), (1 << k) * (lay.B - 1)))
        store.append(H.spread(ctx, x, 1, lay.B))
    return store


def materialize(ctx, lay: Layout, params: list) -> list:
    """저장된 파라미터(압축 모드면 compact) -> forward/backward가 쓰는 store 모양."""
    return expand(ctx, lay, params) if lay.use_compact else params


# ---------------------------------------------------------------- update
def update(ctx, lay: Layout, opt: OptConfig, params: list, grads: list, velocity: list | None):
    """params/velocity는 저장 형태(압축 모드면 compact), grads는 store 모양. 반환: (새 params, 새 velocity 또는 None).
    압축 모드에서는 gradient를 compact로 모아서 momentum/lr/clamp를 compact ciphertext(보통 1개)에서만 한다."""
    eng = ctx.engine
    node_lr = opt.node_lr(lay.n_features)
    vec = (lambda per_kind: lay.compact_vector(lay.store_vector(per_kind))) if lay.use_compact else lay.store_vector
    lr_vec = vec({"threshold": node_lr, "alpha": node_lr, "leaf": opt.leaf_lr(lay.n_classes)})
    if opt.clamp:
        bound = vec({"threshold": opt.threshold_bound, "alpha": opt.softmax_bound, "leaf": opt.softmax_bound})
        inv_r = vec({"threshold": 1 / opt.threshold_radius, "alpha": 1 / opt.softmax_radius, "leaf": 1 / opt.softmax_radius})
    if lay.use_compact:
        grads = compact(ctx, lay, grads)
    new_params, new_vel = [], []
    for q in range(len(params)):
        g = grads[q]
        # 허수부가 iteration을 넘어 쌓이지 않게 한다(he_ops.real_part 참고). Re(x) = (x + conj(x))/2의 1/2은 이미 있는
        # 곱셈(β, lr)에 합쳐서 level을 더 쓰지 않는다: v = β·Re(v_old) + g, p -= lr·Re(v), p = Re(clamp(p)).
        conj2 = lambda ct: eng.add(eng.intt(ct), eng.conjugate(eng.intt(ct), ctx.conjugation_key))   # 2·Re(ct)
        if opt.optimizer == "mom":
            v = g if velocity is None else eng.add(H.mul(ctx, conj2(velocity[q]), opt.beta * 0.5), g)
            v = H.lv(ctx, v)
            new_vel.append(v)
            step = v
        else:
            step = g
        p = eng.subtract(H.lv(ctx, params[q]), H.mul(ctx, conj2(H.lv(ctx, step)), lr_vec[q] * 0.5))
        p = H.he_clamp(ctx, p, bound[q], inv_r[q], real=True) if opt.clamp else H.real_part(ctx, p)
        new_params.append(H.lv(ctx, p))
    return new_params, (new_vel if opt.optimizer == "mom" else None)


# ---------------------------------------------------------------- 평문 에뮬레이션(검증용)
def emulate(X: np.ndarray, Y: np.ndarray, depth: int, spec: ApproxSpec, opt: OptConfig, iterations: int, params0: dict):
    """같은 규칙(reference.compute_gradients poly + GD/momentum + clamp_plain)으로 평문 학습. iteration별 params 반환."""
    from models.gradient_soft_tree.baseline.reference import PolySpec, compute_gradients
    ps = PolySpec(spec.sigmoid_coeffs, 2.0, spec.exp_coeffs, spec.exp_interval, spec.recip_iterations, spec.steepness)
    params = {k: v.copy() for k, v in params0.items()}
    vel = {k: np.zeros_like(v) for k, v in params.items()}
    lrs = {"threshold": opt.node_lr(X.shape[1]), "alpha": opt.node_lr(X.shape[1]), "leaf_logits": opt.leaf_lr(Y.shape[1])}
    bounds = {"threshold": (opt.threshold_bound, 1 / opt.threshold_radius),
              "alpha": (opt.softmax_bound, 1 / opt.softmax_radius), "leaf_logits": (opt.softmax_bound, 1 / opt.softmax_radius)}
    hist = []
    for it in range(iterations):
        grads, _ = compute_gradients(X, Y, params, depth, spec.steepness, ps, balance_lambda=opt.balance)
        for k in params:
            if opt.optimizer == "mom":
                vel[k] = grads[k] if it == 0 else opt.beta * vel[k] + grads[k]
                step = vel[k]
            else:
                step = grads[k]
            params[k] = params[k] - lrs[k] * step
            if opt.clamp:
                params[k] = H.clamp_plain(params[k], *bounds[k])
        hist.append({k: v.copy() for k, v in params.items()})
    return hist
