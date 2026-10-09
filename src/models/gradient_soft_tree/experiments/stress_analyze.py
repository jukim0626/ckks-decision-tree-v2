"""stress_test 결과 분석. python -m models.gradient_soft_tree.experiments.stress_analyze <out_dir>"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from models.gradient_soft_tree.experiments.stress_test import DATA_DEPTH, FAIL, INITS, NEAR, T, data  # noqa: E402


def main():
    d = Path(sys.argv[1])
    rows = [json.loads(l) for l in open(d / "main.jsonl")]
    esc = [json.loads(l) for l in open(d / "escape.jsonl")]
    conds = [(a, b) for a in DATA_DEPTH for b in INITS]
    Ls = {c: min(r["L"] for r in rows if (r["data"], r["init"]) == c) for c in conds}
    L0 = {(r["data"], r["init"], r["seed"]): r["L0"] for r in rows}
    gap = lambda c, L, l0: (L - Ls[c]) / max(l0 - Ls[c], 1e-12)
    by = defaultdict(list)
    for r in rows:
        r["gap"] = gap((r["data"], r["init"]), r["L"], r["L0"])
        yte = data(r["data"])[3]
        r["acc"] = float((np.argmax(r["test_proba"], 1) == yte).mean())
        by[(r["data"], r["init"], r["method"])].append(r)

    print(f"== 1. 실패율(gap >= {FAIL:.0%}) / 평균 test 정확도, seed 64개, T={T} ==")
    meths = ["base", "balance", "noise", "continuation"]
    print(f"{'조건':22s} " + " ".join(f"{m:>15s}" for m in meths) + f" {'multi R=4':>12s} {'R=8':>7s} {'이론 p^8':>8s} {'앙상블8 acc':>11s}")
    agg = defaultdict(list)
    for c in conds:
        cells = []
        for m in meths:
            rs = by[(c[0], c[1], m)]
            f = np.mean([r["gap"] >= FAIL for r in rs])
            agg[m].append((f, np.mean([r["acc"] for r in rs])))
            cells.append(f"{f:5.0%} / {np.mean([r['acc'] for r in rs]):.3f}")
        base = sorted(by[(c[0], c[1], "base")], key=lambda r: r["seed"])
        p = np.mean([r["gap"] >= FAIL for r in base])
        fr = {}
        for R in (4, 8):
            groups = [base[i:i + R] for i in range(0, len(base) - R + 1, R)]
            fr[R] = np.mean([min(r["gap"] for r in g) >= FAIL for g in groups])
        yte = data(c[0])[3]
        ens8 = np.mean([float((np.mean([r["test_proba"] for r in base[i:i + 8]], 0).argmax(1) == yte).mean()) for i in range(0, 64, 8)])
        agg["R4"].append((fr[4], 0)); agg["R8"].append((fr[8], ens8)); agg["p8"].append((p ** 8, 0))
        print(f"{c[0] + '/' + c[1]:22s} " + " ".join(f"{x:>15s}" for x in cells) + f" {fr[4]:12.0%} {fr[8]:7.0%} {p ** 8:8.1%} {ens8:11.3f}")
    print(f"{'평균':22s} " + " ".join(f"{np.mean([a for a, _ in agg[m]]):5.0%} / {np.mean([b for _, b in agg[m]]):.3f}".rjust(15) for m in meths)
          + f" {np.mean([a for a, _ in agg['R4']]):12.0%} {np.mean([a for a, _ in agg['R8']]):7.0%} {np.mean([a for a, _ in agg['p8']]):8.1%}"
          + f" {np.mean([b for _, b in agg['R8']]):11.3f}")

    print(f"\n== 2. 정체된 checkpoint(대책 없는 run 중 실패한 것)에서 처치별 결과 ==")
    start = {(r["data"], r["init"], r["seed"]): r["L"] for r in rows if r["method"] == "base"}
    tr = defaultdict(list)
    for e in esc:
        key = (e["data"], e["init"], e["seed"])
        c = (e["data"], e["init"])
        g = gap(c, e["L"], L0[key])
        tr[e["treat"]].append((e["L"] < start[key] - 1e-4, g < FAIL, g < NEAR))
    n_ck = len({(e["data"], e["init"], e["seed"]) for e in esc})
    print(f"정체 checkpoint {n_ck}개, 각 처치 후 {T} iteration 더 학습")
    print(f"{'처치':12s} {'시도 수':>7s} {'loss 개선':>9s} {'실패 탈출':>9s} {'최선 근처':>9s}")
    for t in ["continue", "reset_v", "balance", "kick0.03", "kick0.1"]:
        v = np.array(tr[t])
        if len(v):
            print(f"{t:12s} {len(v):7d} {v[:, 0].mean():9.0%} {v[:, 1].mean():9.0%} {v[:, 2].mean():9.0%}")
    print("\n조건별 정체 checkpoint 수:", {f"{a}/{b}": len({e['seed'] for e in esc if (e['data'], e['init']) == (a, b)}) for a, b in conds if any((e['data'], e['init']) == (a, b) for e in esc)})


if __name__ == "__main__":
    main()
