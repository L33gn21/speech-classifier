"""학습/평가에 쓰이는 모든 오디오의 log-mel 캐시를 병렬로 미리 생성한다.

dataset 은 lazy 캐싱(첫 접근 시 계산)도 하지만, 첫 epoch 이 여전히 느리다.
이 스크립트로 CPU 코어를 모두 써서 캐시를 warm 해두면 첫 epoch 부터 디스크 로드만 한다.
train.build_splits() 가 실제로 쓰는 파일만 대상으로 한다(불필요한 hifiGAN 전량 캐싱 방지).

실행: .venv/bin/python src/detector/prepare_cache.py
"""
import os
from multiprocessing import Pool

from dataset import load_mel, _cache_path, CACHE_DIR
from train import build_splits


def worker(path):
    try:
        load_mel(path)  # 없으면 계산+저장, 있으면 즉시 반환
        return True
    except Exception as e:
        print(f"FAIL {path}: {e}")
        return False


def main():
    train_files, test_files, holdout_files, _, _ = build_splits()

    # realpath 기준 dedupe (심볼릭 링크로 같은 wav 를 여러 번 가리켜도 1회만)
    seen = {}
    for p, _ in train_files + test_files + holdout_files:
        seen[os.path.realpath(p)] = p
    paths = list(seen.values())

    already = sum(1 for p in paths if os.path.exists(_cache_path(p)))
    print(f"대상 고유 오디오: {len(paths)}개 (이미 캐시됨: {already})")

    nproc = os.cpu_count() or 4
    with Pool(nproc) as pool:
        results = pool.map(worker, paths, chunksize=16)

    ok = sum(results)
    total_bytes = sum(
        os.path.getsize(os.path.join(CACHE_DIR, f))
        for f in os.listdir(CACHE_DIR) if f.endswith(".npy")
    )
    print(f"캐시 완료: {ok}/{len(paths)}  ({nproc} procs)")
    print(f"캐시 디렉터리: {CACHE_DIR}  ({total_bytes/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
