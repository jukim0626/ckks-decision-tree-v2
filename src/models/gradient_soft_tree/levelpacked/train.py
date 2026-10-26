"""levelpacked 학습 오케스트레이터(자신은 GPU를 안 잡고 단계별 worker 프로세스만 띄운다).

python -m models.gradient_soft_tree.levelpacked.train <dataset> <depth> <iterations> <preset>
  preset: gd     = vanilla GD (기존 packed와 같은 알고리즘, lr은 --lr)
          safe   = 제안 방식: momentum 0.9 + lr 1 + 노드 lr F보정(kappa 0.25) + leaf lr K보정 + 합성 다항식 clamp
  옵션: --lr X  --seed N  --level-preset 17
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.runtime.gpu import wait_for_gpu_settle  # noqa: E402
from core.runtime.session import make_session_dir, run_worker_with_oom_retry  # noqa: E402
from core.runtime.subprocess_worker import run_worker_module  # noqa: E402
from models.gradient_soft_tree.levelpacked.tree import OptConfig  # noqa: E402

WORKERS = "models.gradient_soft_tree.levelpacked.workers"
PRESETS = {
    "gd": lambda lr: OptConfig("gd", lr if lr is not None else 2.0),
    "safe": lambda lr: OptConfig("mom", lr if lr is not None else 1.0, 0.9, 0.25, True, leaf_k_scale=True),
}


def _last_json(stdout: str) -> dict:
    return json.loads(stdout.strip().splitlines()[-1])


def _write_environment(path: Path, level_preset: int) -> None:
    """실험 환경(하드웨어/라이브러리/CKKS 파라미터)을 한 번 기록 - 다른 논문과 시간 비교할 때 필요."""
    if path.exists():
        return
    import platform
    import subprocess
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    mem_gb = round(int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal"))) / 1024 ** 2, 1)
    from importlib.metadata import distributions
    desilo = next((f"{d.metadata['Name']} {d.version}" for d in distributions()
                   if d.metadata["Name"] and d.metadata["Name"].lower().startswith("desilofhe")), "unknown")
    env = {"gpu": gpu, "cpu": cpu, "cpu_threads": os.cpu_count(), "ram_gb": mem_gb, "python": platform.python_version(),
           "desilofhe": desilo, "ckks": {"slot_count": 32768, "ring_degree": 65536, "level_preset": level_preset,
                                          "bootstrap": "desilofhe use_bootstrap_to_17_levels" if level_preset == 17 else str(level_preset)},
           "notes": "학습 시간은 iteration별 (gradient + update) 합. 중간 평가(finalize/predict)는 제외."}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, ensure_ascii=False, indent=1))


def _git_commit() -> str:
    """결과 재현용: 실행 시점 코드 버전(작업 트리에 미커밋 변경이 있으면 -dirty)."""
    import subprocess
    root = Path(__file__).resolve().parents[4]
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "src"], cwd=root, capture_output=True, text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _evaluate(session_dir: Path, tag: str, it: int, measures: list, t_start: float, results: Path | None, meta: dict) -> None:
    wait_for_gpu_settle()
    fin = _last_json(run_worker_module(WORKERS, "finalize", str(session_dir)))
    print(f"{tag} [@{it}] finalize: {json.dumps(fin)}", flush=True)
    wait_for_gpu_settle()
    pred = _last_json(run_worker_module(WORKERS, "predict", str(session_dir)))
    print(f"{tag} [@{it}] encrypted inference: {json.dumps(pred)}", flush=True)
    if results is None:
        return
    _write_environment(results.parent / "environment.json", meta.get("level_preset", 17))
    # 세션 폴더는 /tmp라 재부팅 시 사라질 수 있어 iteration별 측정/설정을 results 옆에 복사해 둔다
    keep = results.parent / "sessions" / session_dir.name
    keep.mkdir(parents=True, exist_ok=True)
    for name in ("measurements.jsonl", "config.json"):
        if (session_dir / name).exists():
            shutil.copy2(session_dir / name, keep / name)
    cfg = json.loads((session_dir / "config.json").read_text())
    sizes = {d: sum(f.stat().st_size for f in (session_dir / d).rglob("*") if f.is_file())
             for d in ("keys", "data", "params", "velocity", "client") if (session_dir / d).exists()}
    iter_s = [m["t_grad"] + m["t_update"] for m in measures]
    nodes = cfg["n_internal"] + cfg["n_leaves"] if "n_internal" in cfg else None
    rec_times = {
        "setup": cfg.get("setup_times"),
        "train_iter_sum_s": sum(iter_s),                       # 순수 학습 시간(중간 평가 제외)
        "train_iter_mean_s": sum(iter_s) / len(iter_s),
        "train_iter_steady_mean_s": (sum(iter_s[1:]) / len(iter_s[1:])) if len(iter_s) > 1 else None,  # 첫 iteration 제외
        "grad_mean_s": sum(m["t_grad"] for m in measures) / len(measures),
        "update_mean_s": sum(m["t_update"] for m in measures) / len(measures),
        "per_node_per_iter_s": (sum(iter_s) / len(iter_s) / nodes) if nodes else None,
        "per_level_per_iter_s": sum(iter_s) / len(iter_s) / meta["depth"],
        "train_total_per_node_s": (sum(iter_s) / nodes) if nodes else None,
    }
    rec = {**meta, "iterations": it, "times": rec_times, "sizes_bytes": sizes,
           "n_train": cfg.get("n_train"), "n_test": cfg.get("n_test"), "layout_desc": cfg.get("layout_desc"),
           "n_data_ct": cfg.get("n_data_ct"), "n_param_ct": cfg.get("n_param_ct"), "session_dir": str(session_dir), "train_wall_s": time.time() - t_start,
           "iter_time_mean_s": sum(m["t_grad"] + m["t_update"] for m in measures) / len(measures),
           "bootstrap_per_iter": sum(m["bootstrap_grad"] + m["bootstrap_update"] for m in measures) / len(measures),
           "peak_gpu_mib": max(m["peak_gpu_mib"] for m in measures), "finalize": fin, "predict": pred}
    results.parent.mkdir(parents=True, exist_ok=True)
    with open(results, "a") as f:
        f.write(json.dumps(rec) + "\n")


def train(dataset: str, depth: int, iterations: int, preset: str, lr: float | None, seed: int, level_preset: int,
          fold: int | None = None, results: Path | None = None, checkpoints: list[int] | None = None,
          n_models: int = 1, balance: float = 0.0, bootstrap_key: str = "small", class_weight: str = "",
          scaling: str = "minmax") -> Path:
    """checkpoints: 이 iteration들에서도 finalize + encrypted inference를 해서 결과를 기록(학습은 계속).
    한 번 돌려서 여러 iteration 예산의 결과를 얻기 위함 - 마지막 iteration은 항상 포함."""
    checkpoints = sorted(set((checkpoints or []) + [iterations]))
    opt = PRESETS[preset](lr)
    if balance or class_weight:
        from dataclasses import replace as _replace
        opt = _replace(opt, balance=balance, class_weight=class_weight)
    fold_tag = f"f{fold}" if fold is not None else "s80"
    session_dir = make_session_dir(f"levelpacked_{dataset}_d{depth}_{preset}_{fold_tag}_")
    tag = f"[levelpacked {dataset} d={depth} {preset}]"
    print(f"{tag} opt={asdict(opt)} session={session_dir}", flush=True)
    wait_for_gpu_settle()
    split = f"fold:{fold}" if fold is not None else "0.2"
    run_worker_module(WORKERS, "setup", str(session_dir), dataset, str(depth), str(seed), str(level_preset), split,
                      json.dumps(asdict(opt)), str(n_models), bootstrap_key, scaling)
    t_start = time.time()
    measures = []
    for it in range(1, iterations + 1):
        t0 = time.time()
        out = run_worker_with_oom_retry(lambda: run_worker_module(WORKERS, "iter", str(session_dir)),
                                        epoch=it, log_prefix=tag, max_retries=3)
        m = _last_json(out)
        measures.append(m)
        print(f"{tag} iteration {it}/{iterations} done | {time.time() - t0:.1f}s "
              f"(grad {m['t_grad']:.1f}s + update {m['t_update']:.1f}s, bootstrap {m['bootstrap_grad']}+{m['bootstrap_update']}, "
              f"peak {m['peak_gpu_mib']} MiB)", flush=True)
        if it in checkpoints:
            _evaluate(session_dir, tag, it, measures, t_start, results,
                      {"dataset": dataset, "fold": fold, "depth": depth, "preset": preset, "seed": seed, "opt": asdict(opt),
                       "level_preset": level_preset, "n_models": n_models, "bootstrap_key": bootstrap_key,
                       "balance": balance, "scaling": scaling, "git_commit": _git_commit()})
    return session_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("depth", type=int)
    ap.add_argument("iterations", type=int)
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level-preset", type=int, default=17)
    ap.add_argument("--fold", type=int, default=None, help="evaluation_splits의 fold 번호(논문 실험)")
    ap.add_argument("--results", type=Path, default=None, help="실행 결과를 한 줄씩 쌓을 jsonl")
    ap.add_argument("--checkpoints", type=str, default="", help="중간 평가 iteration, 예: 50,100")
    ap.add_argument("--models", type=str, default="1", help="multi-start 모델 수 R(2의 거듭제곱) 또는 auto(연산량 안 늘리는 최대 R)")
    ap.add_argument("--balance", type=float, default=0.0, help="balance penalty λ")
    ap.add_argument("--class-weight", choices=["", "inv", "sqrt", "cap5", "cap10"], default="",
                    help="가중 MSE: inv(클래스 빈도 역수) / sqrt(그 제곱근) / capN(역수, 평균 1 기준 상한 N), 가중치는 client가 암호화해서 넘김")
    ap.add_argument("--scaling", choices=["minmax", "clip5_95", "clip5_95fb"], default="minmax",
                    help="client 전처리: minmax / 5-95%% 분위수(winsorize) / 분위수가 같은 feature만 min-max")
    ap.add_argument("--bootstrap-key", choices=["small", "regular"], default="small",
                    help="small: 작은 key(~0.26GiB, 느림) / regular: 일반 key(~9.9GiB GPU 메모리, ~3배 빠름)")
    a = ap.parse_args()
    cps = [int(x) for x in a.checkpoints.split(",") if x]
    if a.models == "auto":
        from core.data.dataset import evaluation_splits
        from models.gradient_soft_tree.levelpacked.layout import free_models
        Xtr, _, ytr, yte = evaluation_splits(a.dataset)[a.fold or 0]
        n_models = free_models(len(ytr), Xtr.shape[1], int(max(ytr.max(), yte.max()) + 1), a.depth)
    else:
        n_models = int(a.models)
    train(a.dataset, a.depth, a.iterations, a.preset, a.lr, a.seed, a.level_preset, a.fold, a.results, cps, n_models, a.balance, a.bootstrap_key, a.class_weight, a.scaling)


if __name__ == "__main__":
    main()
