# Speech Accent-by-Country Dataset — Storage & Usage Guide (us-west2 rebuild)

> **Read this first.** This documents everything stored in the project's GCS
> bucket **`gs://qi-ucsd-speech-usw2`** (region **us-west2**). It is the source of
> truth for how the data is laid out, how it was built, and how to use it safely.
> This is a **clean rebuild** — a fresh start from two corpora (GLOBE + Speech
> Accent Archive) plus a separate VoxForge held-out test pool. It does **not**
> share data with the old `gs://qi-ucsd-speech-us` (us-central1) bucket.

- **Project:** `speech-classifier` — classify the country/accent of English speech.
- **Target classes (6):** `US`, `UK`, `CA`, `AU`, `IN`, `CN`
- **Bucket:** `gs://qi-ucsd-speech-usw2` · region **us-west2** · project `qi-ucsd-project`
- **Built:** 2026-07-16 from GLOBE (HuggingFace `MushanW/GLOBE`, CC0) + Speech
  Accent Archive (GMU, CC BY-NC-SA).
- **Ground truth for counts:** the per-class `manifest.csv` files and
  `reports/curation_report.json`. If any number here disagrees, **trust those**.

---

## 1. Bucket layout

```
gs://qi-ucsd-speech-usw2/
├── curated/                       ← USE THIS for training (Standard storage; 5k rebuild, §3)
│   ├── US/  {manifest.csv, audio/(glb_*.flac, saa_*.mp3)}
│   ├── UK/  {manifest.csv, audio/}
│   ├── CA/  {manifest.csv, audio/}
│   ├── AU/  {manifest.csv, audio/}
│   ├── IN/  {manifest.csv, audio/}
│   └── CN/  {manifest.csv, audio/}
├── _archive/curated_v1_8k/        ← previous 8.1k curated pool (superseded, §3)
├── raw/                           ← ARCHIVE / rebuild pool (Standard for now)
│   ├── globe/data/*.parquet       (110 shards, 581,725 clips, 24 kHz, CC0)
│   └── saa/  {audio/saa_*.mp3, saa_manifest.csv}   (2,136 speakers, GMU)
├── test_raw/                      ← TEST-ONLY — never train on this
│   └── voxforge/  {tgz/*.tgz, voxforge_manifest.csv, voxforge_submissions.csv}
├── outputs/classifier/<JOB>/      ← trained models + figures + metrics
└── reports/                       ← machine-readable stats
    ├── globe_report.json  saa_report.json  voxforge_report.json
    └── curation_report.json
```

**Organized by country, not by source** — adding a class is just a new
`curated/<CC>/`. Source is preserved in each manifest and in every audio filename
prefix (`glb_` = GLOBE, `saa_` = SAA), required for channel-leakage analysis and
speaker-disjoint splitting.

---

## 2. The `curated/<CC>/manifest.csv` schema

One row per audio clip. This is the file the training loader reads.

| column   | meaning | example |
|----------|---------|---------|
| `fname`  | filename inside `curated/<CC>/audio/` | `glb_S_000115_106_902.flac` |
| `source` | origin corpus: `GLOBE` or `SAA` | `GLOBE` |
| `speaker`| speaker id, namespaced by source (`glb_<id>` / `saa_<id>`) | `glb_S_000115` |
| `gender` | `F` / `M` / `U` | `F` |
| `age`    | source-dependent band; may be empty | `twenties` |
| `accent` | source-level label kept for provenance | `GLOBE-United States English`, `SAA-mandarin` |

The country label **is the folder** (`US`/`UK`/`CA`/`AU`/`IN`/`CN`), not a column.

---

## 3. What's in each class (actual built counts, 2026-07-16)

| Class | Clips | Speakers | F / M / U | Sources (clips) | Label axis |
|-------|------:|---------:|:---------:|-----------------|------------|
| **US** | 6,243 | 840 | 3,316 / 2,927 / 0 | GLOBE 6,163 · SAA 80 | country (usa) |
| **UK** | 5,933 | 824 | 2,740 / 3,193 / 0 | GLOBE 5,869 · SAA 64 | country (uk) |
| **CA** | 6,002 | 695 | 2,145 / 3,585 / 272 | GLOBE 5,948 · SAA 54 | country (canada) |
| **AU** | 4,825 | 642 | 1,653 / 3,058 / 114 | GLOBE 4,792 · SAA 33 | country (australia) |
| **IN** | 4,064 | 609 | 1,183 / 2,881 / 0 | GLOBE 3,990 · SAA 74 | country (india+pakistan) |
| **CN** | 1,170 | 164 | 349 / 813 / 8 | GLOBE 1,096 (Hong Kong) · SAA 74 | **native_language** (mandarin+cantonese) |

Total: **28,237 clips**. Each `manifest.csv` row count matches these.

> **5k-scale rebuild (2026-07-16, `spec_5k.json`).** These counts supersede the
> original ~8.1k build (US/UK/CA/AU/IN ~1.5k, CN 679). The clip counts were
> too small, so the curation caps were raised — see §4. The old 8.1k pool is
> archived at `gs://qi-ucsd-speech-usw2/_archive/curated_v1_8k/` (rebuildable
> from `raw/` + `_curate/spec.json` anyway). **CN is unchanged (~1.2k, the
> floor):** GLOBE Hong Kong English has only 90 speakers / 1,096 clips total, so
> CN cannot scale with the others without a new source (SpeechOcean762 mainland
> CN — see §5). The imbalance is intentional; training absorbs it with
> macro-F1 + class weighting and can cap at `--per-class` (config default 5000).

- **Audio formats:** GLOBE clips are **FLAC @ 24 kHz**; SAA clips are **mp3**.
  Normalization (16 kHz mono, loudness) is a required preprocessing step — see §6.
- **CN is the smallest class (the floor).** GLOBE has no mainland-China accent
  tag, so CN is built from GLOBE **Hong Kong English** + SAA Mandarin/Cantonese
  L1 speakers. The other five classes were capped generously (~1,500 clips) to
  stay within ~2.2× of CN; training can subsample the big classes to CN's size
  for a fully balanced run (`--per-class`), or rely on macro-F1 + class weighting.

---

## 4. How the curated pool was selected (`curate.py`, hybrid label axis)

Per class, deterministic (`seed=42`), speaker-namespaced, with a global fname
dedupe so no clip lands in two classes:

- **GLOBE** (volume backbone): up to **380 F + 380 M speakers**, **≤20 clips/speaker**,
  gender balanced (5k rebuild caps; the original 8.1k build used 100+100 / ≤15).
  GLOBE clips/speaker are heavy-right-skewed (~8 clips/speaker in a random
  sample), so **speaker count is the real lever** for reaching ~5k/class — not
  clips/speaker. `CN` overrides the clip cap to take all 1,096 of its Hong Kong
  clips (only 90 speakers exist). `IN`/`AU` are speaker-limited (535/609 GLOBE
  speakers available) so they land at ~4k. Matched by exact `accent` string (Common-Voice-style labels —
  see `reports/globe_report.json` for the exact values & counts). Class → accent:
  `US`←"United States English"; `UK`←{"England English","Scottish English",
  "Welsh English"}; `CA`←"Canadian English"; `AU`←"Australian English";
  `IN`←"India and South Asia (India, Pakistan, Sri Lanka)"; `CN`←"Hong Kong English".
- **SAA** (speaker diversity): up to **40 F + 40 M speakers**, 1 clip each.
- **Hybrid label axis:** region classes (US/UK/CA/AU/IN) filter SAA by **birthplace
  country**; `CN` (an L2 class) filters SAA by **native_language** (mandarin,
  cantonese). A speaker matching two classes (e.g. a USA-born Mandarin speaker →
  US by country *and* CN by language) is kept only in the first class in label
  order (region wins), never double-counted.

Rebuild: a short-lived GCE VM (`curate-6class`, us-west2, deleted after) streamed
the 46.5 GB of GLOBE parquet, extracted matching clips, pulled SAA clips from
`raw/saa/`, and wrote `curated/` + `reports/curation_report.json`. The exact
6-class spec used is archived at `gs://qi-ucsd-speech-usw2/_curate/spec.json`;
`classifier/spec_candidates.json` documents the wider candidate set.

---

## 5. Critical caveats — read before training

1. **Channel confound > accent signal.** GLOBE (clean 24 kHz TTS-grade, itself
   derived from Common Voice) and SAA (one read paragraph, GMU) have distinct
   recording fingerprints. Five classes are GLOBE-dominant, so a model can score
   high by "reading the corpus." Mitigate: normalize (§6), keep SAA in the mix,
   and run a channel-leakage probe (train on low-level features only; if classes
   separate, channel is leaking).
2. **Speaker-disjoint splits are mandatory.** Split by the `speaker` column, never
   by clip, or accuracy is inflated by voice memorization. `prepare_data.py` does
   this automatically.
3. **CN is L2 / Hong Kong English, architecturally different** from the five
   native/region classes (US/UK/CA/AU/IN). It is also the smallest class and the
   most male-skewed (254F/417M). Expect CN to behave differently — that is real
   signal, but rule out channel confound (caveat 1) first.
4. **GLOBE ⊂ Common Voice lineage.** If Common Voice is ever added as another
   source, it will overlap GLOBE — keep them disjoint.
5. **NG / JP absent.** Nigeria and Japan are not built: GLOBE has ~no Nigerian or
   Japanese accent data, and their original sources (AfriSpeech, and no free JP
   corpus) were not ingested into this bucket. Re-add AfriSpeech (NG) /
   SpeechOcean762 (mainland CN) if those classes are wanted.

---

## 6. Preprocessing TODO (partly applied at load time)

The curated audio is raw-as-collected. `dataset.py` already resamples to 16 kHz
mono and crops to `MAX_DURATION_S` (8 s) at load time. Still recommended before a
serious run: re-encode to one codec (kills the FLAC-vs-mp3 tell), loudness-
normalize to **−23 LUFS**, and segment SAA's ~20 s paragraph into comparable
windows (GLOBE clips are 0.3–8.7 s).

---

## 7. Held-out test pool — VoxForge (`test_raw/voxforge/`)

> ⚠️ **DO NOT TRAIN ON `test_raw/voxforge/`.** It lives outside `curated/` and
> `raw/` on purpose so training code (which loads `curated/<CC>/`) can never pick
> it up. Keep it that way.

VoxForge English (voxforge.org), archived 2026-07-16: **6,321 submissions /
87,421 clips**, one tgz per speaker with `etc/README` (pronunciation dialect,
gender, age) + `wav/*.wav` + `etc/PROMPTS`. Corpus-, speaker- and channel-disjoint
from GLOBE + SAA, so it is a genuinely unseen evaluation set. `voxforge_manifest.csv`
gives a per-clip index (fname, speaker, gender, age, `accent`=dialect, tgz, member);
`voxforge_submissions.csv` gives per-submission dialects. Dialect distribution
(submissions): American 2,874 · European-L2 828 · British 757 · Canadian 321 ·
Indian 250 · Australian 184 · South African 76 · New Zealand 74 · Irish 33 (+ noise).
To build a labeled test set, match specific dialect strings to the training classes.

---

## 8. Storage classes, cost, licensing

- `curated/`, `reports/`, `outputs/`, `test_raw/` — **Standard** storage (hot).
- `raw/` — kept Standard **for now** (curation reads it often). Transition to
  **Coldline** once the class set is finalized to save storage cost (Coldline has
  a 90-day minimum + retrieval fee — don't do it mid-iteration).
- **Licenses differ per source:** GLOBE = **CC0** (public domain, commercial OK);
  SAA = **CC BY-NC-SA** (non-commercial — the binding constraint on the mixed
  curated pool); VoxForge = **GPL** (test pool, kept separate).

**Upstream sources:** GLOBE = HF `MushanW/GLOBE` (not gated) · SAA = live download
from `accent.gmu.edu/soundtracks/<name>.mp3` (metadata mirror:
`ejlok1/Kaggle-Kernel-Speech-Accent-Archive/speakers_all.csv`) · VoxForge =
`repository.voxforge1.org/.../16kHz_16bit/`.

---

## 9. Quick start

```python
import csv
for cc in ["US", "UK", "CA", "AU", "IN", "CN"]:
    rows = list(csv.DictReader(open(f"curated/{cc}/manifest.csv")))
    # audio at curated/<cc>/audio/<row['fname']>; label = cc
    # SPLIT BY row['speaker'] (speaker-disjoint), then normalize (§6).
```

Download the training set:
```
gcloud storage cp -r gs://qi-ucsd-speech-usw2/curated ./curated
```
