"""여러 preset을 순차로(단일 GPU라 동시 실행 불가) 실행하는 배치 드라이버.

python -m models.gradient_soft_tree.opt.run_sweep iris 3 1 2.0 0 17 recip14 recip8 recip6 recip4 ...
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from models.gradient_soft_tree.opt.train_opt import train  # noqa: E402


def main() -> None:
    dataset_name = sys.argv[1]
    depth = int(sys.argv[2])
    n_epochs = int(sys.argv[3])
    lr = float(sys.argv[4])
    seed = int(sys.argv[5])
    level_preset = int(sys.argv[6])
    presets = sys.argv[7:]

    for idx, preset in enumerate(presets, 1):
        print(f"\n===== [{idx}/{len(presets)}] preset={preset} =====", flush=True)
        t0 = time.time()
        try:
            train(dataset_name, depth, preset, n_epochs, lr, seed, level_preset=level_preset, detailed=False)
        except Exception as e:  # noqa: BLE001
            print(f"[SWEEP ERROR] preset={preset} failed: {e}", flush=True)
        print(f"===== [{idx}/{len(presets)}] preset={preset} done in {time.time()-t0:.1f}s =====", flush=True)


if __name__ == "__main__":
    main()
