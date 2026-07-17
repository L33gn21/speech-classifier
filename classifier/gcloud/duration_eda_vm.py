"""Runs ON the temporary duration-EDA VM (see run_duration_eda.sh).

Lists every curated audio blob, downloads each one to a scratch file, reads
its real duration with ffprobe (works across FLAC/mp3/wav without decoding
the whole file), then builds per-class + overall histograms and an HTML
report. Everything is uploaded back to OUT_PREFIX in GCS; the caller
downloads the small result files and deletes this VM.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import subprocess
import sys
import tempfile

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from google.cloud import storage

BUCKET = sys.argv[1]
OUT_PREFIX = sys.argv[2]  # "reports/duration_eda/<ts>" (no gs://, no bucket)
CLASSES = sys.argv[3].split(",")

client = storage.Client()
bucket = client.bucket(BUCKET)


def probe_one(blob) -> dict | None:
    fname = blob.name.rsplit("/", 1)[-1]
    fd, tmp = tempfile.mkstemp(suffix="_" + fname)
    os.close(fd)
    try:
        blob.download_to_filename(tmp)
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", tmp],
            capture_output=True, text=True, timeout=30,
        )
        seconds = out.stdout.strip()
        if not seconds:
            return None
        return {"class": blob.name.split("/")[1], "fname": fname,
                "seconds": float(seconds), "bytes": blob.size}
    except Exception as exc:
        print(f"! {blob.name}: {exc}", file=sys.stderr)
        return None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


rows = []
for cc in CLASSES:
    blobs = list(client.list_blobs(bucket, prefix=f"curated/{cc}/audio/"))
    blobs = [b for b in blobs if not b.name.endswith("/")]
    print(f"{cc}: probing {len(blobs)} files ...")
    with cf.ThreadPoolExecutor(max_workers=24) as ex:
        for r in ex.map(probe_one, blobs):
            if r:
                rows.append(r)

df = pd.DataFrame(rows)
workdir = tempfile.mkdtemp()
df.to_csv(f"{workdir}/durations.csv", index=False)

summary = df.groupby("class")["seconds"].describe(percentiles=[.5, .9, .95])
summary.to_csv(f"{workdir}/summary.csv")

classes = sorted(df["class"].unique())
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, cc in zip(axes.flat, classes):
    sub = df.loc[df["class"] == cc, "seconds"]
    ax.hist(sub, bins=40, color="#4C72B0", edgecolor="white")
    ax.set_title(f"{cc} (n={len(sub)})")
    ax.set_xlabel("seconds")
for ax in axes.flat[len(classes):]:
    ax.axis("off")
fig.suptitle("Per-clip audio duration by class")
fig.tight_layout()
fig.savefig(f"{workdir}/duration_hist_by_class.png", dpi=140)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.hist(df["seconds"], bins=60, color="#55A868", edgecolor="white")
ax2.set_title(f"All classes combined (n={len(df)})")
ax2.set_xlabel("seconds")
fig2.tight_layout()
fig2.savefig(f"{workdir}/duration_hist_overall.png", dpi=140)

with open(f"{workdir}/report.html", "w", encoding="utf-8") as f:
    f.write("<h1>Audio duration EDA (curated pool)</h1>")
    f.write(summary.to_html())
    f.write('<h2>Overall</h2><img src="duration_hist_overall.png" style="max-width:800px">')
    f.write('<h2>By class</h2><img src="duration_hist_by_class.png" style="max-width:1200px">')

for name in ("durations.csv", "summary.csv", "duration_hist_by_class.png",
             "duration_hist_overall.png", "report.html"):
    bucket.blob(f"{OUT_PREFIX}/{name}").upload_from_filename(f"{workdir}/{name}")

bucket.blob(f"{OUT_PREFIX}/_DONE").upload_from_string(f"rows={len(df)}\n")
print(f"done, {len(df)} clips measured")
