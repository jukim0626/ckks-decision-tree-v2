"""client 전처리 옵션 검증(2026-10-27, CPU, 몇 초).

1) fit_scaler(scaling=clip5_95 / clip5_95fb)가 분위수 공식(combo_study 에뮬레이션과 같은 식)과 같은 값을 내는지
   10개 데이터셋 전 fold의 train/test에서 확인. save_scaler -> load_scaler 왕복 후에도 같아야 한다.
2) class_sample_weights(capN)이 정의(평균 1 역수 가중 -> 상한 N -> 재정규화)대로인지, inv/sqrt와의 관계.

python -m models.gradient_soft_tree.tests.test_scaling_and_weights
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.data.dataset import evaluation_splits, fit_scaler, load_scaler, save_scaler, scale_features  # noqa: E402
from models.gradient_soft_tree.levelpacked import tree as T  # noqa: E402

DATASETS = ["iris", "wine", "breast_cancer", "breast_cancer_nominal", "diabetes", "banknote", "soybean15", "digits",
            "cover10k", "sepsis"]
TOL = 1e-12


def quantile_ref(kind: str, Xtr: np.ndarray, Xev: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(Xtr, 5, axis=0), np.percentile(Xtr, 95, axis=0)
    if kind == "clip5_95fb":
        deg = hi <= lo
        lo, hi = np.where(deg, Xtr.min(0), lo), np.where(deg, Xtr.max(0), hi)
    span = np.where(hi > lo, hi - lo, 1.0)
    return np.clip(2 * (Xev - lo) / span - 1, -1, 1)


def check_scaling() -> None:
    worst = 0.0
    with tempfile.TemporaryDirectory() as d:
        for ds in DATASETS:
            for fold, (Xtr, Xte, _, _) in enumerate(evaluation_splits(ds)):
                for kind in ("clip5_95", "clip5_95fb"):
                    sc = fit_scaler(Xtr, scaling=kind)
                    path = Path(d) / f"{ds}_{fold}_{kind}.json"
                    save_scaler(sc, path, dataset_name=ds)
                    sc2 = load_scaler(path, expected_dataset_name=ds, expected_n_features=Xtr.shape[1])
                    for Xev in (Xtr, Xte):
                        ref = quantile_ref(kind, Xtr, Xev)
                        worst = max(worst, np.abs(scale_features(sc, Xev) - ref).max(), np.abs(scale_features(sc2, Xev) - ref).max())
            # minmax 기본값은 그대로여야 함
            Xtr, Xte, _, _ = evaluation_splits(ds)[0]
            a, b = scale_features(fit_scaler(Xtr), Xte), scale_features(fit_scaler(Xtr, scaling="minmax"), Xte)
            assert np.array_equal(a, b)
    status = "PASS" if worst < TOL else "FAIL"
    print(f"[{status}] 분위수 scaler == 분위수 공식 (10개 데이터셋 전 fold, save/load 왕복 포함) max err={worst:.2e}")
    assert status == "PASS"


def check_weights() -> None:
    rng = np.random.default_rng(0)
    y = rng.choice(4, size=2000, p=[0.6, 0.3, 0.09, 0.01])
    inv, sqrt, cap5 = (T.class_sample_weights(y, 4, k) for k in ("inv", "sqrt", "cap5"))
    prior = np.bincount(y, minlength=4) / len(y)
    ref_inv = (1 / prior)[y] / (1 / prior)[y].mean()
    ref_cap = np.minimum(ref_inv, 5.0)
    ref_cap = ref_cap / ref_cap.mean()
    ok = (np.allclose(inv, ref_inv) and np.allclose(cap5, ref_cap) and abs(cap5.mean() - 1) < 1e-12
          and abs(sqrt.mean() - 1) < 1e-12 and T.class_sample_weights(y, 4, "") is None
          and np.allclose(T.class_sample_weights(y, 4, "cap1000"), inv))  # 상한이 충분히 크면 inv와 같음
    print(f"[{'PASS' if ok else 'FAIL'}] class_sample_weights inv/sqrt/cap5 (w max inv {inv.max():.1f}, cap5 {cap5.max():.2f})")
    assert ok


if __name__ == "__main__":
    check_scaling()
    check_weights()
    print("\n전처리/가중치 검증 통과.")
