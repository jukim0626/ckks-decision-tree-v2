"""CKKS 폭주 재현/분석 전용 스크립트(2026-10-09, wine fold0 it122 폭주). 본 학습 코드(levelpacked/train.py)는 건드리지 않는다.

run    : levelpacked worker(setup/iter)를 그대로 쓰되, 매 iteration 직전 상태(params, velocity)를 snapshots/it_<k>에
         보관한다(최근 KEEP개). worker가 기록하는 health(검증용 decrypt)가 이상하면 멈추고 그 iteration 번호를 남긴다.
replay : 보관된 상태에서 그 iteration 하나를 다시 실행하면서 모든 CKKS 연산의 입력/출력 최대 크기를 decrypt로 기록한다
         (sk 사용 - 분석 전용). 처음으로 비정상 값이 나온 연산과 호출 위치를 찾는다. 상태 파일은 바꾸지 않는다.
fresh  : replay와 같지만 시작 ciphertext(params, velocity, X, Y)를 decrypt 후 새로 암호화해서 실행 - 누적 오차/level·scale
         이력 때문인지 비교.

python -m models.gradient_soft_tree.experiments.repro_explosion run <dataset> <fold> <iterations> <out_dir>
python -m models.gradient_soft_tree.experiments.repro_explosion replay <session_dir> <iteration> [repeats]
python -m models.gradient_soft_tree.experiments.repro_explosion fresh <session_dir> <iteration> [repeats]
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.runtime.session import make_session_dir, run_worker_with_oom_retry  # noqa: E402
from core.runtime.subprocess_worker import run_worker_module  # noqa: E402

WORKERS = "models.gradient_soft_tree.levelpacked.workers"
KEEP = 5
LIMIT = 10.0  # 파라미터가 clamp 상한(2.45)을 크게 넘으면 이상으로 본다


def _git() -> str:
    root = Path(__file__).resolve().parents[4]
    h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    d = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "src"], cwd=root, capture_output=True, text=True).stdout.strip()
    return h + ("-dirty" if d else "")


# ------------------------------------------------------------------ run
def run(dataset: str, fold: int, iterations: int, out_dir: Path) -> None:
    from models.gradient_soft_tree.levelpacked.train import PRESETS
    opt = PRESETS["safe"](None)
    sd = make_session_dir(f"repro_{dataset}_f{fold}_")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"dataset": dataset, "fold": fold, "session_dir": str(sd), "opt": asdict(opt), "git": _git(), "started": time.ctime()}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[repro] session {sd}", flush=True)
    run_worker_module(WORKERS, "setup", str(sd), dataset, "3", "0", "17", f"fold:{fold}", json.dumps(asdict(opt)), "1", "small")
    snaps = sd / "snapshots"
    for it in range(1, iterations + 1):
        snap = snaps / f"it_{it}"          # it번째 iteration을 시작하기 직전 상태
        for d in ("params", "velocity"):
            if (sd / d).exists():
                shutil.copytree(sd / d, snap / d)
        old = snaps / f"it_{it - KEEP}"
        if old.exists():
            shutil.rmtree(old)
        run_worker_with_oom_retry(lambda: run_worker_module(WORKERS, "iter", str(sd)), epoch=it, log_prefix="[repro]", max_retries=3)
        rec = json.loads((sd / "measurements.jsonl").read_text().strip().splitlines()[-1])
        im = _imag_state(sd)
        h = rec.get("health", {})
        hmax = max((h[k] for k in ("alpha", "threshold", "leaf_logits") if k in h), default=float("nan"))
        print(f"[repro] it {it} t={rec['t_total']:.1f}s boot={rec['bootstrap_grad']}+{rec['bootstrap_update']} "
              f"max|param|={hmax:.4g} vel={h.get('velocity_max', float('nan')):.4g} imag(param/vel)={im[0]:.3g}/{im[1]:.3g}", flush=True)
        if not h.get("finite", True) or not (hmax < LIMIT):
            meta.update({"exploded_at": it, "health": h, "snapshot": str(snap)})
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
            print(f"[repro] EXPLODED at it {it} -> replay: repro_explosion replay {sd} {it}", flush=True)
            return
    meta["exploded_at"] = None
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print("[repro] no explosion", flush=True)


def _imag_state(sd: Path) -> tuple:
    """저장된 params/velocity ct의 허수부 최대 |값|(분석 전용, sk 사용). 별도 프로세스로 GPU 메모리 회수."""
    code = (
        "import sys, numpy as np; from pathlib import Path\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})\n"
        "from core.ckks_engine import create_bootstrap_engine\n"
        "from core.data.serialization import load_client_secret_key\n"
        f"sd = Path({str(sd)!r}); e = create_bootstrap_engine(mode='gpu', level_preset=17)\n"
        "sk = load_client_secret_key(e, sd); cj = e.read_conjugation_key(sd / 'keys' / 'conjk.bin')\n"
        "def im(f):\n"
        "    c = e.intt(e.read_ciphertext(f)); d = e.subtract(c, e.conjugate(c, cj))\n"
        "    return float(np.abs(np.real(e.decrypt(e.multiply_imaginary_integer(d, 1), sk)) / 2).max())\n"
        "p = max([im(f) for f in (sd / 'params').glob('*.ct')] or [0.0])\n"
        "v = max([im(f) for f in (sd / 'velocity').glob('*.ct')] or [0.0])\n"
        "print(p, v)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    try:
        a, b = out.stdout.strip().split()[-2:]
        return float(a), float(b)
    except Exception:
        return float("nan"), float("nan")


# ------------------------------------------------------------------ replay
class TracingEngine:
    """모든 ciphertext 연산의 입력/출력 최대 |값|을 decrypt로 기록(분석 전용, 느림)."""
    TRACED = ("add", "subtract", "multiply", "rotate", "evaluate_polynomial", "bootstrap", "merge_bootstrap")

    def __init__(self, engine, sk):
        self._e, self._sk, self.log, self.counts, self.dumps = engine, sk, [], {}, []

    def __getattr__(self, name):
        return getattr(self._e, name)

    def _m(self, x):
        import desilofhe
        if isinstance(x, desilofhe.Ciphertext):
            v = np.real(self._e.decrypt(self._e.intt(x), self._sk))
            return float(np.nanmax(np.abs(v))) if np.isfinite(v).all() else float("inf")
        return None

    def _mi(self, x):
        """허수부 최대 |값| - 실수 데이터만 쓰므로 0이어야 한다."""
        import desilofhe
        if isinstance(x, desilofhe.Ciphertext):
            # decrypt는 실수부만 돌려주므로 Im(x) = -Re(i * (x - conj(x))) / 2 로 꺼낸다
            c = self._e.intt(x)
            d = self._e.subtract(c, self._e.conjugate(c, self.conj_key))
            v = -np.real(self._e.decrypt(self._e.multiply_imaginary_integer(d, 1), self._sk)) / 2
            return float(np.nanmax(np.abs(v))) if np.isfinite(v).all() else float("inf")
        return None

    def _call(self, name, *args, **kw):
        out = getattr(self._e, name)(*args, **kw)
        ins = [m for m in (self._m(a) for a in args[:2]) if m is not None]
        outs = out if isinstance(out, tuple) else (out,)
        st = [f"{f.name}:{f.lineno}" for f in traceback.extract_stack()[:-2]
              if "levelpacked" in f.filename or "ckks_engine" in f.filename][-4:]
        rec = {"i": len(self.log), "op": name, "in_max": max(ins) if ins else None,
               "out_max": max(self._m(o) for o in outs), "level_in": [getattr(a, "level", None) for a in args[:2]],
               "site": " <- ".join(reversed(st))}
        if name == "evaluate_polynomial":
            rec["degree"] = len(args[1]) - 1
        if name in ("evaluate_polynomial", "bootstrap", "merge_bootstrap"):
            rec["in_imag"] = self._mi(args[0])
            rec["out_imag"] = max(self._mi(o) for o in outs)
        if name == "evaluate_polynomial" and rec["out_max"] is not None and rec["out_max"] > 1.5 and len(self.dumps) < 5:
            dec = lambda c: np.real(self._e.decrypt(self._e.intt(c), self._sk))
            self.dumps.append({"i": rec["i"], "x": dec(args[0]), "y": dec(outs[0]), "coeffs": np.array(args[1], dtype=float)})
        self.log.append(rec)
        return out

    def __getattribute__(self, name):
        if name in TracingEngine.TRACED:
            return lambda *a, **k: object.__getattribute__(self, "_call")(name, *a, **k)
        return object.__getattribute__(self, name)


def replay(session_dir: Path, iteration: int, repeats: int = 1, fresh: bool = False) -> None:
    from core.ckks_engine import create_bootstrap_engine
    from core.data.serialization import load_client_secret_key, load_server_context
    from models.gradient_soft_tree.levelpacked import tree as T
    from models.gradient_soft_tree.levelpacked.layout import Layout

    cfg = json.loads((session_dir / "config.json").read_text())
    lay = Layout.from_dict(cfg["layout"])
    opt = T.OptConfig(**cfg["opt"])
    snap = session_dir / "snapshots" / f"it_{iteration}"
    engine = create_bootstrap_engine(mode="gpu", device_id=0, level_preset=cfg["level_preset"])
    sk = load_client_secret_key(engine, session_dir)
    for rep in range(repeats):
        tr = TracingEngine(engine, sk)
        ctx = load_server_context(tr, session_dir / "keys", mode="gpu", device_id=0)
        tr.conj_key = ctx.conjugation_key
        rd = lambda p: engine.read_ciphertext(p)
        X = [[rd(session_dir / "data" / f"X_{g}_{h}.ct") for h in range(lay.H)] for g in range(lay.G)]
        Y = [rd(session_dir / "data" / f"Y_{h}.ct") for h in range(lay.H)]
        params = [rd(snap / "params" / f"p_{q}.ct") for q in range(lay.n_params)]
        velocity = [rd(snap / "velocity" / f"vel_{q}.ct") for q in range(lay.n_params)] if (snap / "velocity").exists() else None
        if fresh:  # 같은 값을 새로 암호화(level/scale/누적 오차 이력 제거)
            renew = lambda ct: engine.encrypt(np.real(engine.decrypt(engine.intt(ct), sk)).tolist(), ctx.pk)
            X = [[renew(c) for c in row] for row in X]
            Y = [renew(c) for c in Y]
            params = [renew(c) for c in params]
            velocity = [renew(c) for c in velocity] if velocity else None
        store = T.materialize(ctx, lay, params)
        grads = T.gradients(ctx, lay, T.default_spec(), X, Y, store, balance=opt.balance)
        n_grad = len(tr.log)
        new_p, new_v = T.update(ctx, lay, opt, params, grads, velocity)
        gmax = max(tr._m(g) for g in grads)
        pmax = max(tr._m(p) for p in new_p)
        bad = next((r for r in tr.log if r["out_max"] is not None and r["out_max"] > 1e3), None)
        tag = f"{'fresh' if fresh else 'replay'}_it{iteration}_rep{rep}"
        out = session_dir / f"{tag}.json"
        for dmp in tr.dumps:   # 이상 출력이 난 다항식 평가의 입력/출력(복호화 값) - 평문 다항식과 비교용
            np.savez(session_dir / f"{tag}_poly{dmp['i']}.npz", **{k: v for k, v in dmp.items() if k != "i"})
        out.write_text(json.dumps({"grad_max": gmax, "param_max": pmax, "n_ops_grad": n_grad, "first_bad": bad,
                                   "log": tr.log}, default=str))
        # 다항식/bootstrap 입력 최대값 요약
        pol = {}
        for r in tr.log:
            if r["op"] == "evaluate_polynomial":
                pol[r["degree"]] = max(pol.get(r["degree"], 0), r["in_max"] or 0)
        boot = max((r["in_max"] or 0) for r in tr.log if "bootstrap" in r["op"])
        print(json.dumps({"tag": tag, "grad_max": gmax, "new_param_max": pmax, "poly_in_max_by_degree": pol,
                          "bootstrap_in_max": boot, "first_bad": bad}, default=str), flush=True)


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "run":
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5]))
    elif cmd in ("replay", "fresh"):
        replay(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1, fresh=(cmd == "fresh"))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
