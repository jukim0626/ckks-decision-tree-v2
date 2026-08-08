"""sign_bootstrap()의 정확도/속도를 재는 미니벤치마크.

fully-encrypted MGI argmin(FindMaxGroupPos 스타일)을 시도하기 전에, 그 핵심 재료인
desilofhe의 sign_bootstrap()이 이 GPU 환경에서 실제로 부호를 정확히 판별하는지,
호출 1번에 얼마나 걸리는지부터 확인한다.
"""

from __future__ import annotations

import time

from desilofhe import Engine


def main() -> None:
    engine = Engine(mode="gpu", use_bootstrap=True, device_id=0)
    print(f"slot_count={engine.slot_count}")

    sk = engine.create_secret_key()
    pk = engine.create_public_key(sk)
    rlk = engine.create_relinearization_key(sk)
    rotk = engine.create_rotation_key(sk)
    conjk = engine.create_conjugation_key(sk)

    t0 = time.time()
    small_bk = engine.create_small_bootstrap_key(sk)
    print(f"create_small_bootstrap_key: {time.time() - t0:.2f}s")

    # 부호 판별 정확도: 다양한 크기의 양수/음수/0 근접값
    test_values = [-5.0, -1.0, -0.1, -0.001, 0.0, 0.001, 0.1, 1.0, 5.0]
    ct = engine.encrypt(test_values, pk)

    t0 = time.time()
    sign_ct = engine.sign_bootstrap(ct, rlk, conjk, rotk, small_bk)
    call_time = time.time() - t0

    decrypted = engine.decrypt(sign_ct, sk)[: len(test_values)]
    print(f"sign_bootstrap 1회 호출: {call_time:.3f}s")
    print("input  -> sign output (expected +-1)")
    for val, out in zip(test_values, decrypted):
        print(f"  {val:8.3f} -> {out.real:+.4f}")

    # FindMaxGroupPos에서 반복 호출될 때의 비용 추정 (candidate 수 3/12/39/90)
    print("\n반복 호출 시간 (level 관리 없이 단순 반복, 실제론 매 호출 level 소모돼서")
    print("bootstrap로 복구해야 하므로 이 숫자는 하한선):")
    n_repeat = 5
    t0 = time.time()
    for _ in range(n_repeat):
        engine.sign_bootstrap(ct, rlk, conjk, rotk, small_bk)
    elapsed = time.time() - t0
    print(f"  {n_repeat}회 평균: {elapsed / n_repeat:.3f}s/call")


if __name__ == "__main__":
    main()
