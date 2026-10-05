"""Pre-generate, in parallel, the log-mel cache for every audio file used in training/evaluation.

The dataset also caches lazily (computed on first access), but the first epoch is still slow.
Warming the cache with this script on all CPU cores makes even the first epoch a pure disk load.
Only the files train.build_splits() actually uses are processed (avoids caching all of hifiGAN needlessly).

Run: .venv/bin/python src/detector/prepare_cache.py
"""
import os
from multiprocessing import Pool

from dataset import load_mel, _cache_path, CACHE_DIR
from train import build_splits


def worker(path):
    try:
        load_mel(path)  # computes+saves if missing, returns immediately if present
        return True
    except Exception as e:
        print(f"FAIL {path}: {e}")
        return False


def main():
    train_files, test_files, holdout_files, _, _ = build_splits()

    # dedupe by realpath (only once even if symlinks point to the same wav several times)
    seen = {}
    for p, _ in train_files + test_files + holdout_files:
        seen[os.path.realpath(p)] = p
    paths = list(seen.values())

    already = sum(1 for p in paths if os.path.exists(_cache_path(p)))
    print(f"unique target audio files: {len(paths)} (already cached: {already})")

    nproc = os.cpu_count() or 4
    with Pool(nproc) as pool:
        results = pool.map(worker, paths, chunksize=16)

    ok = sum(results)
    total_bytes = sum(
        os.path.getsize(os.path.join(CACHE_DIR, f))
        for f in os.listdir(CACHE_DIR) if f.endswith(".npy")
    )
    print(f"cache done: {ok}/{len(paths)}  ({nproc} procs)")
    print(f"cache directory: {CACHE_DIR}  ({total_bytes/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
