"""Runs ON the temporary ASVspoof-ingest VM (see ingest_asvspoof.sh).

Streams the HuggingFace dataset ``Bisher/ASVspoof_2019_LA`` (ASVspoof 2019 LA —
the bonafide-vs-spoof anti-spoofing corpus, 16 kHz WAV) and writes it into the
project bucket in the curated schema, PRESERVING the original protocol splits:

    curated_spoof/asvspoof2019_la/<split>/manifest.csv
    curated_spoof/asvspoof2019_la/<split>/audio/asv_<audio_file_name>.wav

    HF split      -> our split   notes
    train         -> train       attacks A01-A06
    validation    -> dev         attacks A01-A06, speaker-disjoint from train
    test          -> eval        attacks A07-A19  (UNSEEN synthesis systems)

The protocol split is load-bearing: ``eval`` uses synthesis systems that never
appear in train/dev, so a random re-split would leak attack types and inflate
spoof metrics — the whole point of ASVspoof is generalization to unseen attacks.
Speakers are disjoint across splits by protocol design too.

We store the raw 16 kHz WAV bytes as-is (``Audio(decode=False)`` — no decode /
re-encode, lossless) and leave loudness/codec normalization to the training
loader so that real and fake receive the identical transform (channel-confound
control, DATASET.md 5).

Manifest schema (one row per clip):
    fname,source,speaker,key,system_id,split
"""
# 임시 ASVspoof 인제스트 VM 위에서 실행된다 (ingest_asvspoof.sh 참고).
# HuggingFace 데이터셋을 스트리밍으로 받아, 원본 WAV 바이트를 재인코딩 없이
# 버킷의 curated_spoof/asvspoof2019_la/<split>/ 에 그대로 쓴다.
# 원본 프로토콜 스플릿(train/dev/eval)을 반드시 보존한다 — eval 은 train/dev 에
# 없는 합성 시스템(A07~A19)을 쓰므로 랜덤 재분할하면 공격 유형이 누수된다.
from __future__ import annotations

import collections
import concurrent.futures as cf
import csv
import io
import json
import sys

from datasets import Audio, load_dataset
from google.cloud import storage

BUCKET = sys.argv[1]
DEST_ROOT = sys.argv[2]          # e.g. "curated_spoof/asvspoof2019_la" (no gs://, no bucket)
HF_DATASET = sys.argv[3] if len(sys.argv) > 3 else "Bisher/ASVspoof_2019_LA"

# HF split name -> our protocol split folder name
SPLIT_MAP = {"train": "train", "validation": "dev", "test": "eval"}
FIELDS = ["fname", "source", "speaker", "key", "system_id", "split"]
MAX_INFLIGHT = 512               # bound the upload backlog so memory stays flat

client = storage.Client()
bucket = client.bucket(BUCKET)


def process_split(hf_split: str, out_split: str) -> list[dict]:
    """Stream one HF split and mirror it into curated_spoof/<out_split>/."""
    ds = load_dataset(HF_DATASET, split=hf_split, streaming=True)
    # decode=False -> audio arrives as {"bytes": <raw wav>, "path": ...}; no decode.
    ds = ds.cast_column("audio", Audio(decode=False))
    key_feat = ds.features["key"]                       # ClassLabel(bonafide, spoof)
    audio_prefix = f"{DEST_ROOT}/{out_split}/audio"
    rows: list[dict] = []

    def upload(fname: str, data: bytes) -> None:
        bucket.blob(f"{audio_prefix}/{fname}").upload_from_string(
            data, content_type="audio/wav"
        )

    n = 0
    inflight: set = set()
    with cf.ThreadPoolExecutor(max_workers=32) as ex:
        for row in ds:
            base = row["audio_file_name"]
            fname = f"asv_{base}.wav"
            data = row["audio"]["bytes"]
            if data is None:
                print(f"! {out_split}: {base} has no bytes; skipped", file=sys.stderr)
                continue
            key = row["key"]
            key = key_feat.int2str(key) if isinstance(key, int) else key
            rows.append({
                "fname": fname,
                "source": "ASVspoof2019LA",
                "speaker": f"asv_{row['speaker_id']}",
                "key": key,
                "system_id": row["system_id"],
                "split": out_split,
            })
            inflight.add(ex.submit(upload, fname, data))
            n += 1
            if len(inflight) >= MAX_INFLIGHT:
                done, inflight = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
                for f in done:
                    f.result()                          # surface upload errors
            if n % 5000 == 0:
                print(f"[{out_split}] {n} uploaded ...", flush=True)
        for f in cf.as_completed(inflight):
            f.result()

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
    bucket.blob(f"{DEST_ROOT}/{out_split}/manifest.csv").upload_from_string(
        buf.getvalue(), content_type="text/csv"
    )
    print(f"[{out_split}] DONE n={n}", flush=True)
    return rows


def breakdown(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "key": dict(collections.Counter(r["key"] for r in rows)),
        "system_id": dict(collections.Counter(r["system_id"] for r in rows)),
        "speakers": len({r["speaker"] for r in rows}),
    }


all_rows: list[dict] = []
for hf_split, out_split in SPLIT_MAP.items():
    all_rows.extend(process_split(hf_split, out_split))

by_split = collections.defaultdict(list)
for r in all_rows:
    by_split[r["split"]].append(r)

report = {
    "dest": f"gs://{BUCKET}/{DEST_ROOT}",
    "hf_dataset": HF_DATASET,
    "schema": FIELDS,
    "total": len(all_rows),
    "by_split": {s: breakdown(rows) for s, rows in by_split.items()},
}
bucket.blob(f"{DEST_ROOT}/ingest_report.json").upload_from_string(
    json.dumps(report, indent=2), content_type="application/json"
)
bucket.blob(f"{DEST_ROOT}/_DONE").upload_from_string(
    json.dumps({s: len(r) for s, r in by_split.items()}) + "\n"
)
print("ALL DONE", report["by_split"], flush=True)
