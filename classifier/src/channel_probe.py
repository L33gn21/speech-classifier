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
# 채널 누수 프로브 (DATASET.md §5.1).
#
# 묻는 것: 분류기의 성능 중 얼마가 "억양"이 아니라 "녹음 채널/코퍼스 지문"에서
# 나오는가? 6개 중 5개 클래스가 GLOBE(깨끗한 24kHz FLAC) 우세이고 GLOBE 안에서도
# 억양별로 녹음 집단(마이크)이 다르므로, 모델이 억양이 아니라 "코퍼스/마이크를
# 읽어" 점수를 낼 수 있다. 이 confound 가 v3 가 미지 코퍼스(VoxForge)에서 CA→US
# 로 붕괴한 유력 원인이라, 이를 직접 측정한다.
#
# 방법(§5.1): 채널/마이크/코덱 정보는 담되 음소(발음) 내용은 거의 없는 "저수준
# 음향 특징"으로, 실제 모델과 "동일한 화자분리 분할"에서 단순 선형 프로브(로지스틱
# 회귀)를 학습한다. 이 특징만으로 6개국이 우연 이상으로 갈리면 채널이 새는 것이다.
#   - lowlevel: 장기평균스펙트럼 + 스펙트럼형태 통계 + 고역대 에너지비 + 노이즈 플로어.
#   - silence : 가장 조용한 프레임들의 로그-멜 스펙트럼만. 침묵은 억양을 담을 수
#               없으므로, 이것이 클래스를 가르면 "명백한" 채널 누수다(가장 깨끗한 격리).
#   - US↔CA 이진 프로브: 회귀가 난 바로 그 쌍. 저수준 특징이 US/CA 를 50% 훨씬
#               넘게 가르면 붕괴에 채널 원인이 있는 것.
#   - GLOBE↔SAA 소스 프로브: 양성 대조군. 코덱/샘플레이트가 다르니 특징이 이를
#               거의 완벽히 갈라야 정상 — 못 가르면 특징이 약한 것이라 국가 수치도 무의미.
#
# 화자분리는 의도적/보수적이다: 한 화자의 마이크 암기를 금지하므로 남는 분리력은
# "클래스 단위" 채널 편향 — 미지 코퍼스로 전이 안 되는 바로 그 confound 다.
#
# CPU 전용(librosa + scikit-learn; torch/GPU 불필요). config.CURATED_ROOT 에서
# 오디오를 읽고 config.OUTPUT_DIR(Vertex 에선 AIP_MODEL_DIR→버킷)에 JSON 판정을 쓴다.
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
# 저수준 특징 추출 (채널 우세, 음소 비의존)
# ---------------------------------------------------------------------------
# STFT/멜 파라미터: 25ms 창 / 10ms 홉 / 40 멜밴드 (음성 표준). 특징은 프레임 축으로
# 통계를 내(mean/std/percentile) 발화 내용(음소 시퀀스)은 평균으로 씻겨나가고
# 녹음 채널의 주파수 응답·대역폭·노이즈 특성만 남게 만든다.
_N_FFT = 400
_HOP = 160
_N_MELS = 40
_HF_BANDS = 8       # 상위 멜밴드 수(코덱 저역통과/대역폭 tell)
_SILENCE_PCTL = 15  # 이 백분위 이하 에너지 프레임을 "침묵"으로 간주

# 각 특징 그룹의 차원 (matrix 조립·자기검증에 사용).
_GROUP_DIMS = {
    "ltas": 2 * _N_MELS,   # 장기평균 로그멜 mean+std
    "shape": 12,           # centroid/bw/rolloff/flatness/zcr/rms 의 mean+std
    "hf": 1,               # 고역대 에너지비 평균
    "floor": 2,            # 프레임 에너지(dB) 5·10 백분위 = 노이즈 플로어
    "silence": _N_MELS + 1,  # 침묵 프레임 로그멜 평균 + 침묵 레벨(dB)
}
# 프로브별로 어떤 그룹을 쓰는지.
PROBE_FEATURES = {
    "lowlevel": ["ltas", "shape", "hf", "floor"],  # 일반 저수준(약간의 억양 가능)
    "silence": ["silence"],                         # 엄격: 억양 불가능
}


def _load_audio(path: Path, max_seconds: float) -> np.ndarray | None:
    """Load an audio file as float32 mono @ SAMPLE_RATE, capped to max_seconds.

    librosa-only (no torch) so the probe stays a lightweight CPU job. Handles
    both GLOBE FLAC and SAA mp3 via soundfile / audioread(ffmpeg). Returns None
    on decode failure (the clip is skipped, not fatal).
    """
    # torch 없이 librosa 로만 로드해 프로브를 가벼운 CPU 잡으로 유지한다.
    # FLAC(GLOBE)·mp3(SAA) 모두 처리. 디코드 실패 시 None(해당 클립만 건너뜀).
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
    # 파형에서 채널 우세 저수준 특징을 그룹별로 계산해 반환한다. 모든 그룹은
    # 시간축 통계라 음소 시퀀스는 씻겨나가고 녹음 채널 특성만 남는다.
    import librosa

    # STFT 크기 스펙트럼(D)을 한 번 계산해 형태 통계에 재사용하고, 그 파워로부터
    # 멜 스펙트럼을 만든다. spectral_flatness/rms 는 멜이 아니라 STFT 크기(선형
    # 주파수 bin)를 요구하므로 D 를 넘겨야 한다.
    D = np.abs(librosa.stft(wav, n_fft=_N_FFT, hop_length=_HOP))    # [1+n_fft/2, T]
    D = np.maximum(D, 1e-10)
    S = librosa.feature.melspectrogram(
        S=D ** 2, sr=SAMPLE_RATE, n_mels=_N_MELS)                  # [n_mels, T] power
    S = np.maximum(S, 1e-10)
    logS = librosa.power_to_db(S)                  # [n_mels, T]
    frame_pow = (D ** 2).sum(axis=0)               # [T] per-frame energy
    frame_db = librosa.power_to_db(np.maximum(frame_pow, 1e-10))  # [T]

    # -- LTAS: 장기평균 로그멜 스펙트럼(mean)+변동(std). 마이크 주파수응답·코덱
    #    저역통과가 지배(음소는 평균으로 상쇄). --
    ltas = np.concatenate([logS.mean(axis=1), logS.std(axis=1)])   # [2*n_mels]

    # -- 스펙트럼 형태 통계: 대역폭/코덱 tell. 각 프레임 스칼라의 mean+std. --
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

    # -- 고역대 에너지비: 상위 멜밴드/전체. mp3 저역통과 vs FLAC 광대역 tell. --
    hf = np.array([float(np.mean(S[-_HF_BANDS:].sum(axis=0) / (frame_pow + 1e-10)))],
                  dtype=np.float32)                               # [1]

    # -- 노이즈 플로어: 프레임 에너지(dB)의 5·10 백분위. --
    floor = np.array([float(np.percentile(frame_db, 5)),
                      float(np.percentile(frame_db, 10))], dtype=np.float32)  # [2]

    # -- 침묵 프레임 로그멜: 가장 조용한 프레임들(하위 _SILENCE_PCTL%)의 평균 로그멜.
    #    침묵은 억양을 담을 수 없으므로 순수 채널/마이크 노이즈 색채다. --
    thr = np.percentile(frame_db, _SILENCE_PCTL)
    sil_idx = np.where(frame_db <= thr)[0]
    if sil_idx.size < 3:  # 거의 무음 없는 짧은 클립 — 가장 조용한 3프레임 사용
        sil_idx = np.argsort(frame_db)[:3]
    silence = np.concatenate([
        logS[:, sil_idx].mean(axis=1),                 # [n_mels]
        [float(frame_db[sil_idx].mean())],             # 침묵 레벨(dB)
    ]).astype(np.float32)                              # [n_mels+1]

    return {"ltas": ltas.astype(np.float32), "shape": shape,
            "hf": hf, "floor": floor, "silence": silence}


def _featurize_path(args: tuple[str, int, str, str, float]):
    """joblib worker: load one clip and featurize it. Returns (feats, label, ...)."""
    # joblib 워커: 한 클립을 로드·특징화. (특징, 라벨, source, country) 반환.
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
# 특징 행렬 조립 + 프로브
# ---------------------------------------------------------------------------
def _matrix(rows: list[dict], keys: list[str]) -> np.ndarray:
    """Stack selected feature groups from a list of per-clip feature dicts."""
    # 클립별 특징 dict 리스트에서 선택한 그룹들을 이어붙여 행렬로 만든다.
    return np.vstack([np.concatenate([r[k] for k in keys]) for r in rows])


def _extract_split(df, curated_root: str, max_seconds: float, n_jobs: int,
                   tag: str):
    """Featurize every clip in a split. Returns (feats_list, labels, sources, countries)."""
    # 한 분할의 모든 클립을 특징화한다. 실패 클립은 제외한다.
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
    # 표준화 + 로지스틱 회귀 프로브를 학습하고 test 분할 지표를 반환한다.
    # class_weight='balanced' 로 불균형(특히 source 프로브)을 보정한다.
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

    # 우연 기준선: test 라벨 분포를 따르는 무작위 예측(stratified dummy).
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
# 판정
# ---------------------------------------------------------------------------
# v3(WavLM) 실제 모델의 내부 test macro-F1 (참조점; reports/2026-07-17-...md).
V3_TEST_MACRO_F1 = 0.624


def _severity(silence_mf1: float, chance_mf1: float) -> str:
    """Grade leakage mainly from the SILENCE probe (accent-impossible, so cleanest)."""
    # 누수 등급은 주로 침묵 프로브로 매긴다(억양이 불가능하므로 가장 깨끗한 신호).
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
    # 분할 생성 → 특징화 → 전 프로브 실행 → 리포트 dict 반환.
    # 실제 모델과 동일한 화자분리 분할을 쓴다(train 으로 프로브 학습, test 로 평가).
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
    #    소스가 둘 다 있는 경우에만(대개 SAA 가 소수라도 존재). 특징이 채널을
    #    실제로 잡는지 검증 — 거의 완벽해야 정상.
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
# 자기검증 (오디오/GCS 불필요) — 특징 파이프라인이 도는지 확인
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
    # 짧은 클립(패딩 경로)과 조립 함수도 확인.
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
