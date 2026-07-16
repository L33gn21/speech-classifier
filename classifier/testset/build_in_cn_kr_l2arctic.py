#!/usr/bin/env python
"""Build an UNSEEN held-out test set for IN/CN/KR from L2-ARCTIC (NathanRoll HF mirror).

- Source corpus disjoint from all training sources (CommonVoice/GLOBE/VCTK/EdAcc/
  Svarah/AfriSpeech/SpeechOcean762/SAA) -> genuine unseen channel + speakers.
- Speaker-disjoint by construction (distinct corpus). Per-speaker clip cap applied.
- Uploads to gs://qi-ucsd-speech-us/test_heldout/<CC>/  (NOT under curated/ -> never trained on).
"""
import io, os, csv, random, subprocess, sys
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

random.seed(42)
OUT = os.path.join(os.path.dirname(__file__), "l2arctic_out")
BUCKET = "gs://qi-ucsd-speech-us/test_heldout"
CAP_PER_SPEAKER = 150

# L2-ARCTIC speaker code -> (class, L1 name, gender) from official L2-ARCTIC v5 docs.
SPK = {
    # Hindi -> IN
    "ASI": ("IN", "Hindi", "M"), "RRBI": ("IN", "Hindi", "M"),
    "SVBI": ("IN", "Hindi", "M"), "TNI": ("IN", "Hindi", "F"),
    # Mandarin -> CN
    "BWC": ("CN", "Mandarin", "M"), "LXC": ("CN", "Mandarin", "F"),
    "NCC": ("CN", "Mandarin", "F"), "TXHC": ("CN", "Mandarin", "M"),
    # Korean -> KR
    "HJK": ("KR", "Korean", "F"), "HKK": ("KR", "Korean", "M"),
    "YDCK": ("KR", "Korean", "F"), "YKWK": ("KR", "Korean", "M"),
}

def sniff_ext(b: bytes) -> str:
    if b[:4] == b"RIFF": return "wav"
    if b[:4] == b"fLaC": return "flac"
    if b[:4] == b"OggS": return "ogg"
    if b[:3] == b"ID3" or b[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"): return "mp3"
    return "wav"

def main():
    print("[1/5] downloading L2-ARCTIC parquet shards ...", flush=True)
    shards = [hf_hub_download("NathanRoll/l2-arctic-dataset",
                              f"data/train-0000{i}-of-00002.parquet",
                              repo_type="dataset") for i in (0, 1)]
    # collect rows per target speaker
    rows_by_spk = {}
    for sh in shards:
        t = pq.read_table(sh)
        d = t.to_pydict()
        n = len(d["speaker"])
        for i in range(n):
            spk = d["speaker"][i]
            if spk not in SPK:
                continue
            rows_by_spk.setdefault(spk, []).append({
                "audio": d["audio"][i], "file_id": d["file_id"][i],
                "transcript": d.get("transcript", [None]*n)[i],
                "sr": d.get("sample_rate", [None]*n)[i],
            })
    print("[2/5] speakers found:", {k: len(v) for k, v in sorted(rows_by_spk.items())}, flush=True)

    manifests = {}  # cc -> list of manifest rows
    for spk, rows in rows_by_spk.items():
        cc, l1, gender = SPK[spk]
        random.shuffle(rows)
        keep = rows[:CAP_PER_SPEAKER]
        adir = os.path.join(OUT, cc, "audio")
        os.makedirs(adir, exist_ok=True)
        for r in keep:
            a = r["audio"]
            b = a["bytes"] if isinstance(a, dict) else a
            if b is None:
                continue
            ext = sniff_ext(b)
            fname = f"l2arctic_{spk}_{r['file_id']}.{ext}"
            with open(os.path.join(adir, fname), "wb") as f:
                f.write(b)
            manifests.setdefault(cc, []).append({
                "fname": fname, "source": "L2Arctic", "speaker": f"l2arctic_{spk}",
                "gender": gender, "age": "", "accent": f"L2Arctic-{l1}",
            })

    print("[3/5] writing manifests ...", flush=True)
    for cc, mrows in manifests.items():
        mrows.sort(key=lambda x: x["fname"])
        mpath = os.path.join(OUT, cc, "manifest.csv")
        with open(mpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["fname","source","speaker","gender","age","accent"])
            w.writeheader(); w.writerows(mrows)
        print(f"   {cc}: {len(mrows)} clips, {len(set(r['speaker'] for r in mrows))} speakers", flush=True)

    print("[4/4] LOCAL BUILD COMPLETE. Verify then upload manually.", flush=True)
    print("OUT_DIR:", OUT, flush=True)

if __name__ == "__main__":
    main()
