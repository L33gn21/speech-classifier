"""Channel-leakage probe (DATASET.md §5.1).

Question this answers: *how much of the classifier's power comes from the
recording channel / corpus fingerprint rather than from accent?* Five of the six
classes are GLOBE-dominant (clean 24 kHz FLAC) with CN mixing GLOBE Hong Kong +
SAA (mp3), and even within GLOBE each accent was recorded by a different Common
Voice population — so a model can score high by "reading the corpus/mic" instead
of the accent. That confound is the leading suspect for the v3 CA→US collapse on
VoxForge (a *different* channel), which is why we measure it directly.

Method (the §5.1 recipe): train a *simple linear probe* on **low-level acoustic
features** that carry channel/mic/codec information but essentially no phonetic
content, using the **same speaker-disjoint splits** the real model uses. If those
features let a logistic regression separate the six countries well above chance,
the channel is leaking. Two probes, from general to strict:

  - ``lowlevel`` — long-term average spectrum + spectral-shape stats + high-band
    energy ratio + noise floor. General channel/mic/codec signature.
  - ``silence``  — the log-mel spectrum of the *quietest* frames only. Silence
    cannot carry an accent, so if this separates the classes it is *unambiguous*
    channel leakage (the cleanest possible isolation).

Plus two targeted cuts:

  - ``US↔CA`` binary probe — the exact pair that regresses. If low-level features
    separate US from CA far above 50%, the collapse has a channel explanation.
  - ``GLOBE↔SAA`` source probe — a *positive control*. These are literally
    different codecs/sample rates, so the features MUST separate them near-
    perfectly; if they don't, the features are too weak and the country numbers
    mean nothing.

Speaker-disjoint is deliberate and conservative: it forbids the probe from
memorizing a *single speaker's* mic, so any remaining separability is a
*class-level* channel bias — precisely the confound that fails to transfer to an
unseen corpus.

Runs on CPU only (librosa + scikit-learn; no torch, no GPU). Reads audio from
``config.CURATED_ROOT`` (local dir or FUSE-mounted GCS on Vertex) and writes a
JSON verdict to ``config.OUTPUT_DIR`` (``AIP_MODEL_DIR`` on Vertex → the bucket).

Example (Vertex CPU custom job): ``gcloud/submit_probe_job.sh --per-class=1500``
Local:                           ``python channel_probe.py --per-class=300``
Self-test (no audio needed):     ``python channel_probe.py --selftest``
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from config import (
    CURATED_ROOT,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    MAX_DURATION_S,
    OUTPUT_DIR,
    SAMPLE_RATE,
    SEED,
)
from prepare_data import build_splits, report

# ---------------------------------------------------------------------------
# Low-level feature extraction (channel-dominant, phonetics-agnostic)
# ---------------------------------------------------------------------------
# STFT/mel parameters: 25 ms window / 10 ms hop / 40 mel bands (speech standard). The
# features are statistics over the frame axis (mean/std/percentile), so the utterance
# content (phoneme sequence) averages out and only the recording channel's frequency
# response, bandwidth and noise characteristics remain.
_N_FFT = 400
_HOP = 160
_N_MELS = 40
_HF_BANDS = 8       # number of top mel bands (codec low-pass / bandwidth tell)
_SILENCE_PCTL = 15  # frames at or below this energy percentile count as "silence"

# Dimension of each feature group (used for matrix assembly and the self-test).
_GROUP_DIMS = {
    "ltas": 2 * _N_MELS,   # long-term average log-mel mean+std
    "shape": 12,           # mean+std of centroid/bw/rolloff/flatness/zcr/rms
    "hf": 1,               # mean high-band energy ratio
    "floor": 2,            # 5th/10th percentile of frame energy (dB) = noise floor
    "silence": _N_MELS + 1,  # mean log-mel of silent frames + silence level (dB)
}
# Which groups each probe uses.
PROBE_FEATURES = {
    "lowlevel": ["ltas", "shape", "hf", "floor"],  # general low-level (some accent information possible)
    "silence": ["silence"],                         # strict: cannot carry accent
}


def _load_audio(path: Path, max_seconds: float) -> np.ndarray | None:
    """Load an audio file as float32 mono @ SAMPLE_RATE, capped to max_seconds.

    librosa-only (no torch) so the probe stays a lightweight CPU job. Handles
    both GLOBE FLAC and SAA mp3 via soundfile / audioread(ffmpeg). Returns None
    on decode failure (the clip is skipped, not fatal).
    """
    import librosa

    try:
        wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True,
                              duration=max_seconds)
    except Exception:
        return None
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size < _N_FFT:  # too short to STFT — pad so librosa doesn't error
        wav = np.pad(wav, (0, _N_FFT - wav.size))
    if not np.isfinite(wav).all():
        wav = np.nan_to_num(wav)
    return wav


def featurize(wav: np.ndarray) -> dict[str, np.ndarray]:
    """Compute channel-dominant low-level features from a waveform.

    Returns a dict of named feature groups (see ``_GROUP_DIMS``). All groups are
    frame-statistics (mean/std/percentile over time) so the phoneme *sequence*
    averages out and what remains is the recording's channel signature.
    """
    import librosa

    # Compute the STFT magnitude spectrum (D) once and reuse it for the shape statistics;
    # the mel spectrum is built from its power. spectral_flatness/rms need the STFT
    # magnitude (linear frequency bins), not mel, so D has to be passed in.
    D = np.abs(librosa.stft(wav, n_fft=_N_FFT, hop_length=_HOP))    # [1+n_fft/2, T]
    D = np.maximum(D, 1e-10)
    S = librosa.feature.melspectrogram(
        S=D ** 2, sr=SAMPLE_RATE, n_mels=_N_MELS)                  # [n_mels, T] power
    S = np.maximum(S, 1e-10)
    logS = librosa.power_to_db(S)                  # [n_mels, T]
    frame_pow = (D ** 2).sum(axis=0)               # [T] per-frame energy
    frame_db = librosa.power_to_db(np.maximum(frame_pow, 1e-10))  # [T]

    # -- LTAS: long-term average log-mel spectrum (mean) + variation (std). Dominated by the
    #    microphone frequency response and the codec low-pass (phonemes cancel out). --
    ltas = np.concatenate([logS.mean(axis=1), logS.std(axis=1)])   # [2*n_mels]

    # -- Spectral-shape statistics: bandwidth/codec tell. mean+std of each per-frame scalar. --
    def _ms(x):
        x = np.asarray(x, dtype=np.float64).ravel()
        return [float(np.mean(x)), float(np.std(x))]

    cent = librosa.feature.spectral_centroid(S=D, sr=SAMPLE_RATE)
    bw = librosa.feature.spectral_bandwidth(S=D, sr=SAMPLE_RATE)
    roll = librosa.feature.spectral_rolloff(S=D, sr=SAMPLE_RATE, roll_percent=0.85)
    flat = librosa.feature.spectral_flatness(S=D)
    zcr = librosa.feature.zero_crossing_rate(wav, frame_length=_N_FFT, hop_length=_HOP)
    rms = librosa.feature.rms(S=D, frame_length=_N_FFT, hop_length=_HOP)
    shape = np.array(_ms(cent) + _ms(bw) + _ms(roll) + _ms(flat)
                     + _ms(zcr) + _ms(rms), dtype=np.float32)      # [12]

    # -- High-band energy ratio: top mel bands / total. mp3 low-pass vs. FLAC wideband tell. --
    hf = np.array([float(np.mean(S[-_HF_BANDS:].sum(axis=0) / (frame_pow + 1e-10)))],
                  dtype=np.float32)                               # [1]

    # -- Noise floor: 5th/10th percentile of the frame energy (dB). --
    floor = np.array([float(np.percentile(frame_db, 5)),
                      float(np.percentile(frame_db, 10))], dtype=np.float32)  # [2]

    # -- Silent-frame log-mel: mean log-mel of the quietest frames (lowest _SILENCE_PCTL%).
    #    Silence cannot carry an accent, so this is pure channel/microphone noise color. --
    thr = np.percentile(frame_db, _SILENCE_PCTL)
    sil_idx = np.where(frame_db <= thr)[0]
    if sil_idx.size < 3:  # short clip with almost no silence — use the 3 quietest frames
        sil_idx = np.argsort(frame_db)[:3]
    silence = np.concatenate([
        logS[:, sil_idx].mean(axis=1),                 # [n_mels]
        [float(frame_db[sil_idx].mean())],             # silence level (dB)
    ]).astype(np.float32)                              # [n_mels+1]

    return {"ltas": ltas.astype(np.float32), "shape": shape,
            "hf": hf, "floor": floor, "silence": silence}


def _featurize_path(args: tuple[str, int, str, str, float]):
    """joblib worker: load one clip and featurize it. Returns (feats, label, ...)."""
    path, label, source, country, max_seconds = args
    wav = _load_audio(Path(path), max_seconds)
    if wav is None:
        return None
    try:
        feats = featurize(wav)
    except Exception:
        return None
    return feats, int(label), str(source), str(country)


# ---------------------------------------------------------------------------
# Feature-matrix assembly + probing
# ---------------------------------------------------------------------------
def _matrix(rows: list[dict], keys: list[str]) -> np.ndarray:
    """Stack selected feature groups from a list of per-clip feature dicts."""
    return np.vstack([np.concatenate([r[k] for k in keys]) for r in rows])


def _extract_split(df, curated_root: str, max_seconds: float, n_jobs: int,
                   tag: str):
    """Featurize every clip in a split. Returns (feats_list, labels, sources, countries)."""
    # Clips that fail to load are excluded.
    from joblib import Parallel, delayed

    jobs = []
    for _, row in df.iterrows():
        p = os.path.join(curated_root, row["country"], "audio", row["filename"])
        jobs.append((p, row["label"], row.get("source", ""), row["country"],
                     max_seconds))
    print(f"[{tag}] featurizing {len(jobs)} clips on {n_jobs} job(s)...")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_featurize_path)(j) for j in jobs)
    results = [r for r in results if r is not None]
    feats = [r[0] for r in results]
    labels = np.array([r[1] for r in results])
    sources = np.array([r[2] for r in results])
    countries = np.array([r[3] for r in results])
    print(f"[{tag}] featurized {len(feats)}/{len(jobs)} "
          f"({len(jobs) - len(feats)} decode failures)")
    return feats, labels, sources, countries


def _fit_probe(Xtr, ytr, Xte, yte, *, multiclass: bool):
    """Fit a standardized logistic-regression probe; return metrics on the test split."""
    # class_weight='balanced' corrects for the imbalance (especially in the source probe).
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced",
                           C=1.0, random_state=SEED),
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    # Chance baseline: random predictions following the test label distribution (stratified dummy).
    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy.fit(Xtr, ytr)
    dpred = dummy.predict(Xte)

    labels_present = sorted(set(ytr.tolist()) | set(yte.tolist()))
    avg = "macro"
    out = {
        "accuracy": float(accuracy_score(yte, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average=avg,
                                   labels=labels_present, zero_division=0)),
        "chance_accuracy": float(accuracy_score(yte, dpred)),
        "chance_macro_f1": float(f1_score(yte, dpred, average=avg,
                                          labels=labels_present, zero_division=0)),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
    }
    if multiclass:
        per = f1_score(yte, pred, average=None, labels=list(range(len(LABELS))),
                       zero_division=0)
        out["per_class_f1"] = {LABELS[i]: float(per[i]) for i in range(len(LABELS))}
        out["confusion_matrix"] = confusion_matrix(
            yte, pred, labels=list(range(len(LABELS)))).tolist()
    return out


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
# Internal test macro-F1 of the actual v3 (WavLM) model (reference; reports/2026-07-17-...md).
V3_TEST_MACRO_F1 = 0.624


def _severity(silence_mf1: float, chance_mf1: float) -> str:
    """Grade leakage mainly from the SILENCE probe (accent-impossible, so cleanest)."""
    lift = silence_mf1 - chance_mf1
    if silence_mf1 >= 0.45 or lift >= 0.30:
        return "SEVERE"
    if silence_mf1 >= 0.30 or lift >= 0.18:
        return "MODERATE"
    if lift >= 0.08:
        return "MINOR"
    return "CLEAN"


def _print_confusion(cm: list[list[int]]) -> None:
    print("    confusion (rows=true, cols=pred):")
    print("            " + "".join(f"{l:>8s}" for l in LABELS))
    for i, r in enumerate(cm):
        print(f"    {LABELS[i]:>6s}  " + "".join(f"{v:8d}" for v in r))


def run(curated_root: str, per_class: int, max_seconds: float, n_jobs: int) -> dict:
    """Build splits, featurize, run all probes, and return the report dict."""
    # Uses the same speaker-disjoint splits as the real model (fit on train, evaluate on test).
    train_df, _val_df, test_df = build_splits(
        curated_root=curated_root, per_class=per_class, seed=SEED)
    report("probe-train", train_df)
    report("probe-test", test_df)

    tr = _extract_split(train_df, curated_root, max_seconds, n_jobs, "train")
    te = _extract_split(test_df, curated_root, max_seconds, n_jobs, "test")
    tr_feats, tr_y, tr_src, tr_cc = tr
    te_feats, te_y, te_src, te_cc = te

    probes: dict = {}

    # -- 1) 6-class country probe: lowlevel & silence --
    for name, keys in PROBE_FEATURES.items():
        Xtr, Xte = _matrix(tr_feats, keys), _matrix(te_feats, keys)
        probes[f"country_6class__{name}"] = _fit_probe(
            Xtr, tr_y, Xte, te_y, multiclass=True)

    # -- 2) US↔CA binary probe (the regressed pair) --
    us, ca = LABEL2ID["US"], LABEL2ID["CA"]
    for name, keys in PROBE_FEATURES.items():
        mtr = np.isin(tr_y, [us, ca])
        mte = np.isin(te_y, [us, ca])
        Xtr = _matrix([tr_feats[i] for i in np.where(mtr)[0]], keys)
        Xte = _matrix([te_feats[i] for i in np.where(mte)[0]], keys)
        probes[f"US_vs_CA__{name}"] = _fit_probe(
            (Xtr), (tr_y[mtr] == ca).astype(int),
            (Xte), (te_y[mte] == ca).astype(int), multiclass=False)

    # -- 3) GLOBE↔SAA source probe (positive control) --
    #    Only when both sources are present (SAA usually exists, even if as a minority).
    #    Verifies that the features really capture the channel — near-perfect is expected.
    def _src_bin(s):
        return np.array([1 if str(x).upper() == "SAA" else 0 for x in s])
    ytr_src, yte_src = _src_bin(tr_src), _src_bin(te_src)
    if ytr_src.sum() >= 5 and yte_src.sum() >= 5:
        Xtr, Xte = _matrix(tr_feats, PROBE_FEATURES["lowlevel"]), \
            _matrix(te_feats, PROBE_FEATURES["lowlevel"])
        probes["source_GLOBE_vs_SAA__lowlevel"] = _fit_probe(
            Xtr, ytr_src, Xte, yte_src, multiclass=False)
    else:
        probes["source_GLOBE_vs_SAA__lowlevel"] = {
            "skipped": "too few SAA clips in the sampled splits"}

    # -- verdict --
    sil = probes["country_6class__silence"]
    low = probes["country_6class__lowlevel"]
    severity = _severity(sil["macro_f1"], sil["chance_macro_f1"])
    report_dict = {
        "config": {
            "per_class": per_class, "max_seconds": max_seconds,
            "seed": SEED, "feature_groups": _GROUP_DIMS,
            "v3_test_macro_f1_ref": V3_TEST_MACRO_F1,
        },
        "probes": probes,
        "verdict": {
            "severity": severity,
            "silence_6class_macro_f1": sil["macro_f1"],
            "silence_6class_chance_macro_f1": sil["chance_macro_f1"],
            "lowlevel_6class_macro_f1": low["macro_f1"],
            "lowlevel_fraction_of_v3": round(low["macro_f1"] / V3_TEST_MACRO_F1, 3),
            "us_ca_lowlevel_balanced_acc": probes["US_vs_CA__lowlevel"].get(
                "balanced_accuracy"),
            "source_control_balanced_acc": probes[
                "source_GLOBE_vs_SAA__lowlevel"].get("balanced_accuracy"),
        },
    }
    return report_dict


def _print_report(rep: dict) -> None:
    v = rep["verdict"]
    print("\n" + "=" * 68)
    print("CHANNEL-LEAKAGE PROBE — VERDICT")
    print("=" * 68)
    for name, p in rep["probes"].items():
        if "skipped" in p:
            print(f"\n[{name}]  SKIPPED — {p['skipped']}")
            continue
        print(f"\n[{name}]  n_train={p['n_train']} n_test={p['n_test']}")
        print(f"    macro_f1 = {p['macro_f1']:.3f}   (chance {p['chance_macro_f1']:.3f})")
        print(f"    accuracy = {p['accuracy']:.3f}   bal_acc {p['balanced_accuracy']:.3f}"
              f"   (chance {p['chance_accuracy']:.3f})")
        if "per_class_f1" in p:
            print("    per-class F1: " + "  ".join(
                f"{k} {val:.2f}" for k, val in p["per_class_f1"].items()))
        if "confusion_matrix" in p:
            _print_confusion(p["confusion_matrix"])
    print("\n" + "-" * 68)
    print(f"  SEVERITY: {v['severity']}")
    print(f"  silence-only 6-class macro-F1 : {v['silence_6class_macro_f1']:.3f} "
          f"(chance {v['silence_6class_chance_macro_f1']:.3f})  "
          f"← accent-impossible; any lift = pure channel")
    print(f"  low-level 6-class macro-F1    : {v['lowlevel_6class_macro_f1']:.3f} "
          f"= {v['lowlevel_fraction_of_v3']:.0%} of v3's real 0.624")
    print(f"  US↔CA low-level balanced acc  : {v['us_ca_lowlevel_balanced_acc']}  "
          f"(0.50 = no channel shortcut for the regressed pair)")
    print(f"  source GLOBE↔SAA control      : {v['source_control_balanced_acc']}  "
          f"(should be ≈1.0 — confirms features capture channel)")
    print("-" * 68)
    print("  Read: high silence/low-level macro-F1 ≫ chance ⇒ the model can score")
    print("  by reading the corpus/mic, not the accent — mitigate via DATASET.md")
    print("  §6 (one codec + loudness norm) and re-measure. High US↔CA here pins")
    print("  the VoxForge CA→US collapse on channel, not genuine accent difficulty.")
    print("=" * 68)


# ---------------------------------------------------------------------------
# self-test (no audio / no GCS needed) — validates the feature pipeline
# ---------------------------------------------------------------------------
def _selftest() -> int:
    rng = np.random.default_rng(0)
    dims_ok = True
    for _ in range(3):
        wav = rng.standard_normal(int(SAMPLE_RATE * 2.0)).astype(np.float32) * 0.1
        f = featurize(wav)
        for k, d in _GROUP_DIMS.items():
            if f[k].shape != (d,):
                print(f"  ! group {k}: got {f[k].shape}, want ({d},)")
                dims_ok = False
        if not all(np.isfinite(f[k]).all() for k in f):
            print("  ! non-finite feature value")
            dims_ok = False
    # Also check a short clip (the padding path) and the assembly function.
    short = rng.standard_normal(50).astype(np.float32)
    fs = featurize(np.pad(short, (0, _N_FFT)))
    M = _matrix([featurize(wav), fs], PROBE_FEATURES["lowlevel"])
    exp_cols = sum(_GROUP_DIMS[k] for k in PROBE_FEATURES["lowlevel"])
    if M.shape != (2, exp_cols):
        print(f"  ! matrix shape {M.shape}, want (2, {exp_cols})")
        dims_ok = False
    print("selftest:", "OK" if dims_ok else "FAILED",
          f"(lowlevel dim={exp_cols}, silence dim={_GROUP_DIMS['silence']})")
    return 0 if dims_ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Channel-leakage probe (DATASET.md §5.1)")
    ap.add_argument("--curated-root", default=str(CURATED_ROOT))
    ap.add_argument("--per-class", type=int, default=1500,
                    help="clips/class cap for the probe (smaller = faster)")
    ap.add_argument("--max-seconds", type=float, default=MAX_DURATION_S,
                    help="crop each clip to this many seconds (matches the model view)")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="parallel featurization workers (-1 = all cores)")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--selftest", action="store_true",
                    help="run the offline feature-pipeline self-test and exit")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())

    rep = run(args.curated_root, args.per_class, args.max_seconds, args.n_jobs)
    _print_report(rep)

    os.makedirs(args.output_dir, exist_ok=True)
    dest = os.path.join(args.output_dir, "channel_probe_report.json")
    with open(dest, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
