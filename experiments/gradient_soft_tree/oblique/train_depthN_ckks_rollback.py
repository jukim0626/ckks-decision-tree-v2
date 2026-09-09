"""train_depthN_ckks.py에 위험 감지 + 롤백 메커니즘을 추가한 버전.

**동기 (2026-09-09, EXPERIMENT_LOG/[[project_ckks_oblique_gate]] 참고)**: 오블리크 gate가
CKKS에서 반복적으로 발산하는 원인을 다항식 차수(degree)와 weight 정규화(weight_decay)로
막으려 했지만 둘 다 근본 해결이 아니었음 - plaintext에서는 peak|z|가 4.9~6.6 정도로만
커지는데 실제 CKKS는 139302까지 튀는 케이스가 관찰됨(iris depth=2, lr=6.0). 이건 "z가
자연스럽게 크다"는 문제가 아니라, **CKKS 안에서 bootstrap이 한 번 값을 오염시키면 다음
epoch의 gradient가 그 오염을 그대로 증폭시키는 피드백 루프**로 재해석됨. 다항식은
수학적으로 어떤 유한 구간 밖에서도 결국 발산하므로 "완전히 안전한 다항식"은 애초에
불가능 - 그래서 이 버전은 "안 터지게 막기"가 아니라 **"터지는 사건이 다음 epoch으로
전파되기 전에 감지해서 되돌리기"**를 시도한다.

**메커니즘**: 매 epoch 끝나면 check_worker_N.py로 max|z|를 관찰(기존과 동일). 안전
(<=danger_threshold)하면 그 epoch의 params를 checkpoint로 저장하고 다음 epoch으로 진행.
위험하면 checkpoint(직전 안전 상태)로 params를 복원하고, 그 epoch만 lr을 절반으로 낮춰서
재시도(최대 max_retries회). 그래도 안전해지지 않으면 이 epoch은 포기하고 checkpoint
상태 그대로 다음 epoch으로 넘어간다(오염된 파라미터를 그대로 들고 가는 것보다 안전).

baseline `train_depthN_ckks.py`는 절대 안 건드림 - setup/epoch/check/finalize worker를
그대로 재사용하되(`epoch_worker_N.py`에 lr override 인자만 하위호환으로 추가됨),
오케스트레이션 로직만 이 새 파일에 둔다.

사용법: python -m experiments.gradient_soft_tree.oblique.train_depthN_ckks_rollback \
        iris 2 30 6.0 0 17 --danger-threshold 2.0 --max-retries 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO_ROOT = Path(__file__).resolve().parents[3]


def _gpu_memory_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    return int(out[0]) if out else 0


def _wait_for_gpu_settle(threshold_mib: int = 500, timeout_s: float = 60.0, poll_s: float = 1.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if _gpu_memory_used_mib() < threshold_mib:
            return
        time.sleep(poll_s)


def _run(module: str, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"{module} 실패:\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    return result.stdout


def train(
    dataset_name: str,
    depth: int,
    n_epochs: int,
    lr: float,
    seed: int,
    level_preset: int | None = 17,
    max_train: int | None = None,
    danger_threshold: float = 2.0,
    max_retries: int = 3,
    perturb_std: float = 0.05,
) -> dict:
    session_dir = Path(tempfile.mkdtemp(prefix=f"oblique_rollback_depth{depth}_"))
    params_dir = session_dir / "params"
    checkpoint_dir = session_dir / "checkpoint"

    print(
        f"[setup] dataset={dataset_name} depth={depth} epochs={n_epochs} lr={lr} seed={seed} "
        f"level_preset={level_preset} max_train={max_train} danger_threshold={danger_threshold} "
        f"max_retries={max_retries}",
        flush=True,
    )
    _wait_for_gpu_settle()
    _run(
        "experiments.gradient_soft_tree.oblique.setup_worker_N",
        str(session_dir), dataset_name, str(depth), str(seed), str(lr),
        str(level_preset) if level_preset is not None else "none",
        str(max_train) if max_train is not None else "none",
    )
    # epoch 0(초기화 직후)를 최초의 "안전한 checkpoint"로 저장.
    shutil.copytree(params_dir, checkpoint_dir)

    n_rollbacks = 0
    n_epochs_given_up = 0

    for epoch in range(1, n_epochs + 1):
        attempt_lr = lr
        for retry in range(max_retries + 1):
            _wait_for_gpu_settle()
            t0 = time.time()
            _run(
                "experiments.gradient_soft_tree.oblique.epoch_worker_N",
                str(session_dir), str(attempt_lr),
            )
            elapsed = time.time() - t0

            _wait_for_gpu_settle()
            check_stdout = _run(
                "experiments.gradient_soft_tree.oblique.check_worker_N", str(session_dir), str(epoch)
            )
            check = json.loads(check_stdout.strip().splitlines()[-1])
            safe = check["max_z"] <= danger_threshold

            if safe:
                shutil.rmtree(checkpoint_dir)
                shutil.copytree(params_dir, checkpoint_dir)
                print(
                    f"[oblique-rollback {dataset_name} depth={depth}] epoch {epoch}/{n_epochs} done "
                    f"(lr={attempt_lr:.4g}, retry={retry}) | {elapsed:.1f}s | "
                    f"max|z|={check['max_z']:.3f} max|w|={check['max_w_abs']:.3f} max|b|={check['max_b_abs']:.3f}",
                    flush=True,
                )
                break

            if retry < max_retries:
                n_rollbacks += 1
                attempt_lr = attempt_lr / 2.0
                print(
                    f"[oblique-rollback {dataset_name} depth={depth}] epoch {epoch}/{n_epochs} "
                    f"⚠️ max|z|={check['max_z']:.3f} > {danger_threshold} - 롤백 후 lr={attempt_lr:.4g}로 재시도"
                    f"({retry + 1}/{max_retries})",
                    flush=True,
                )
                shutil.rmtree(params_dir)
                shutil.copytree(checkpoint_dir, params_dir)
            else:
                n_epochs_given_up += 1
                shutil.rmtree(params_dir)
                shutil.copytree(checkpoint_dir, params_dir)
                # 2026-09-09 "막힘" 대응: 그냥 checkpoint로 되돌리기만 하면 다음 epoch이
                # 똑같은 lr=lr 첫 시도부터 결정론적으로 똑같이 실패하는 게 실측 확인됨
                # (iris depth=2 lr=6.0, epoch13/14/15가 완전히 동일한 값으로 반복 실패).
                # perturb_worker_N으로 checkpoint의 w/b에 작은 랜덤 섭동을 더해서
                # 다음 epoch이 다른 지점에서 출발하게 만든다.
                _wait_for_gpu_settle()
                _run(
                    "experiments.gradient_soft_tree.oblique.perturb_worker_N",
                    str(session_dir), str(seed * 10000 + epoch), str(perturb_std),
                )
                shutil.rmtree(checkpoint_dir)
                shutil.copytree(params_dir, checkpoint_dir)
                print(
                    f"[oblique-rollback {dataset_name} depth={depth}] epoch {epoch}/{n_epochs} "
                    f"⚠️⚠️ {max_retries}회 재시도해도 max|z|={check['max_z']:.3f} 위험 - 이 epoch 포기, "
                    f"직전 안전 상태에 섭동(std={perturb_std}) 추가 후 유지",
                    flush=True,
                )

    _wait_for_gpu_settle()
    stdout = _run("experiments.gradient_soft_tree.oblique.finalize_worker_N", str(session_dir), str(n_epochs))
    result = json.loads(stdout.strip().splitlines()[-1])
    result["n_rollbacks"] = n_rollbacks
    result["n_epochs_given_up"] = n_epochs_given_up
    print(
        f"[oblique-rollback {dataset_name} depth={depth}] max abs diff vs plaintext = {result['max_err']:.5f} | "
        f"train_acc={result['train_acc']:.4f} test_acc={result['test_acc']:.4f} | "
        f"n_rollbacks={n_rollbacks} n_epochs_given_up={n_epochs_given_up}",
        flush=True,
    )
    print(f"[oblique-rollback {dataset_name} depth={depth}] session_dir={session_dir}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("depth", type=int)
    parser.add_argument("n_epochs", type=int)
    parser.add_argument("lr", type=float)
    parser.add_argument("seed", type=int)
    parser.add_argument("level_preset")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--danger-threshold", type=float, default=2.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--perturb-std", type=float, default=0.05)
    args = parser.parse_args()
    level_preset = None if args.level_preset == "none" else int(args.level_preset)
    train(
        args.dataset_name, args.depth, args.n_epochs, args.lr, args.seed,
        level_preset=level_preset, max_train=args.max_train,
        danger_threshold=args.danger_threshold, max_retries=args.max_retries,
        perturb_std=args.perturb_std,
    )


if __name__ == "__main__":
    main()
