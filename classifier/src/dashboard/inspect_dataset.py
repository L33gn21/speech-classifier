"""Dataset visualization report for the curated speech-accent corpus.

Reads `curated/<CC>/manifest.csv` (schema: fname, source, speaker, gender,
age, accent — see classifier/DATASET.md §2) for each target class and writes
a single self-contained HTML report with charts to REPORT_OUT.

This module also backs `serve_dataset_report.py`, a small local web server
with a "Refresh" button that re-reads the manifests on demand.

Usage:
    python inspect_dataset.py
    python inspect_dataset.py --root gs://qi-ucsd-speech-us/curated
    python inspect_dataset.py --root ../curated --classes US UK IN KR

Live dashboard (refresh button, no need to re-run manually):
    python serve_dataset_report.py
"""
# 정제된(curated) 억양 음성 데이터셋을 위한 시각화 리포트 생성 모듈.
#
# 각 타깃 클래스별로 `curated/<국가코드>/manifest.csv`
# (스키마: fname, source, speaker, gender, age, accent — 자세한 내용은
# classifier/DATASET.md §2 참조)를 읽어들여, 차트들이 포함된 하나의
# 독립적인(self-contained) HTML 리포트를 REPORT_OUT 경로에 생성한다.
#
# 이 모듈은 serve_dataset_report.py(새로고침 버튼이 있는 로컬 웹 서버)에서도
# 그대로 재사용된다. CLI로 1회성 정적 파일만 만들 수도, 서버로 띄워서
# 버튼 클릭마다 다시 읽어오게 할 수도 있다.
#
# 사용 예시:
#     python inspect_dataset.py
#     python inspect_dataset.py --root gs://qi-ucsd-speech-us/curated
#     python inspect_dataset.py --root ../curated --classes US UK IN KR
#
# 실시간 대시보드(수동 재실행 없이 버튼으로 갱신):
#     python serve_dataset_report.py
from __future__ import annotations

import argparse
import base64
import datetime
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 출력 없는 환경(서버/CI)에서도 그림을 그릴 수 있도록 백엔드 고정
import matplotlib.pyplot as plt
import pandas as pd

# 대시보드 카드 톤에 맞춘 차트 팔레트/스타일. 흰 카드 위에 얹히므로 배경은
# 투명, 축/그리드는 옅게 — 화면(CSS)과 이미지(matplotlib)가 같은 디자인처럼
# 보이도록 통일한다.
_ACCENT = "#6366f1"
_PALETTE = ["#6366f1", "#22c55e", "#f59e0b", "#ec4899", "#06b6d4", "#8b5cf6", "#ef4444", "#84cc16"]
_INK = "#0f172a"
_MUTED = "#475569"
_GRID = "#e2e8f0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.edgecolor": _GRID,
    "axes.labelcolor": _MUTED,
    "axes.titlecolor": _INK,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.titlepad": 14,
    "axes.labelsize": 15,
    "axes.labelweight": "bold",
    "axes.grid": True,
    "grid.color": _GRID,
    "grid.linewidth": 0.8,
    "xtick.color": _INK,
    "ytick.color": _INK,
    "xtick.labelsize": 14,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "text.color": _INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.axisbelow": True,
})

DEFAULT_ROOT = "gs://qi-ucsd-speech-us/curated"
DEFAULT_CLASSES = ["US", "UK", "IN", "NG", "CA", "JP", "CN", "AU", "KR"]
# fake(합성음성) 탐지용 spoof 코퍼스 — 국가 curated/와 다른 프리픽스/스키마
# (fname,source,speaker,key,system_id,split). DATASET.md §10 참조.
DEFAULT_SPOOF_ROOT = "gs://qi-ucsd-speech-usw2/curated_spoof/asvspoof2019_la"
SPOOF_SPLITS = ["train", "dev", "eval"]
# 최종 HTML 리포트가 저장될 기본 경로 (classifier/reports/dataset_report.html)
# 이 파일은 classifier/src/dashboard/ 아래에 있으므로 3단계 위가 classifier/.
REPORT_OUT = Path(__file__).resolve().parent.parent.parent / "reports" / "dataset_report.html"

_gcs_client = None  # lazy-init — Cloud Run/로컬 모두 google-cloud-storage 하나로 통일


def _gcs():
    # google-cloud-storage 클라이언트를 지연 생성한다. 인증은 ADC(Application
    # Default Credentials)를 쓴다: Cloud Run에서는 서비스 계정으로 자동 처리되고,
    # 로컬에서는 `gcloud auth application-default login` 한 번이면 된다.
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage

        _gcs_client = storage.Client()
    return _gcs_client


def read_manifest(root: str, cc: str) -> pd.DataFrame | None:
    # 특정 클래스(국가 코드, 예: "US")의 manifest.csv를 읽어온다.
    # root가 gs:// 로 시작하면 google-cloud-storage로 읽고, 로컬 경로면
    # 그냥 파일을 연다. 파일이 없으면 None을 반환.
    path = f"{root.rstrip('/')}/{cc}/manifest.csv"
    if root.startswith("gs://"):
        bucket_name, _, prefix = root[len("gs://"):].partition("/")
        blob_path = f"{prefix}/{cc}/manifest.csv" if prefix else f"{cc}/manifest.csv"
        blob = _gcs().bucket(bucket_name).blob(blob_path)
        if not blob.exists():
            print(f"  ! {cc}: no manifest at {path}")
            return None
        return pd.read_csv(io.StringIO(blob.download_as_text()))
    local = Path(path)
    if not local.exists():
        print(f"  ! {cc}: no manifest at {local}")
        return None
    return pd.read_csv(local)


def collect_manifests(root: str, classes: list[str]) -> dict[str, pd.DataFrame]:
    # classes에 있는 모든 클래스의 manifest.csv를 읽어 {클래스: DataFrame} 딕셔너리로 모은다.
    # 서버 모드에서 "새로고침" 버튼을 누를 때마다 이 함수가 다시 호출된다.
    print(f"Reading manifests from {root} ...")
    dfs: dict[str, pd.DataFrame] = {}
    for cc in classes:
        df = read_manifest(root, cc)
        if df is not None and len(df):
            dfs[cc] = df
            print(f"  {cc}: {len(df)} clips, {df['speaker'].nunique()} speakers")
    return dfs


def read_spoof_manifest(root: str, split: str) -> pd.DataFrame | None:
    # curated_spoof/<split>/manifest.csv 를 읽는다. 스키마:
    # fname,source,speaker,key,system_id,split (key=bonafide/spoof).
    path = f"{root.rstrip('/')}/{split}/manifest.csv"
    if root.startswith("gs://"):
        bucket_name, _, prefix = root[len("gs://"):].partition("/")
        blob_path = f"{prefix}/{split}/manifest.csv" if prefix else f"{split}/manifest.csv"
        blob = _gcs().bucket(bucket_name).blob(blob_path)
        if not blob.exists():
            print(f"  ! spoof/{split}: no manifest at {path}")
            return None
        return pd.read_csv(io.StringIO(blob.download_as_text()))
    local = Path(path)
    if not local.exists():
        print(f"  ! spoof/{split}: no manifest at {local}")
        return None
    return pd.read_csv(local)


def collect_spoof_manifests(root: str, splits: list[str]) -> dict[str, pd.DataFrame]:
    # splits(train/dev/eval)의 spoof manifest.csv를 모두 읽어 {split: DataFrame}으로 모은다.
    print(f"Reading spoof manifests from {root} ...")
    dfs: dict[str, pd.DataFrame] = {}
    for split in splits:
        df = read_spoof_manifest(root, split)
        if df is not None and len(df):
            dfs[split] = df
            n_bona = int((df["key"].str.lower() == "bonafide").sum())
            print(f"  {split}: {len(df)} clips, {n_bona} bonafide, {len(df) - n_bona} spoof, "
                  f"{df['speaker'].nunique()} speakers")
    return dfs


def chart_spoof_composition(spoof_dfs: dict[str, pd.DataFrame]) -> str:
    # 스플릿별 bonafide(real)/spoof(fake) 클립 수를 누적 막대그래프로 표시.
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    splits = list(spoof_dfs.keys())
    bona = [int((spoof_dfs[s]["key"].str.lower() == "bonafide").sum()) for s in splits]
    spoof = [len(spoof_dfs[s]) - b for s, b in zip(splits, bona)]
    ax.bar(splits, bona, label="bonafide (real)", color=_PALETTE[1], width=0.5, zorder=3)
    ax.bar(splits, spoof, bottom=bona, label="spoof (fake)", color=_PALETTE[6], width=0.5, zorder=3)
    for i, s in enumerate(splits):
        ax.text(i, bona[i] + spoof[i], f"{bona[i] + spoof[i]:,}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=_INK)
    ax.set_ylabel("clips")
    ax.set_title("Spoof corpus: bonafide vs spoof per split")
    ax.legend(frameon=False)
    return fig_to_data_uri(fig)


def chart_spoof_attack_systems(spoof_dfs: dict[str, pd.DataFrame]) -> str:
    # 스플릿별 spoof 공격 시스템(system_id) 분포를 보여준다 — train/dev는 A01-A06,
    # eval은 A07-A19(미지 공격)로 겹치지 않아야 정상(프로토콜 스플릿 보존 여부 육안 확인용).
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    splits = list(spoof_dfs.keys())
    all_systems = sorted({sid for df in spoof_dfs.values()
                           for sid in df.loc[df["key"].str.lower() == "spoof", "system_id"].unique()})
    bottom = [0] * len(splits)
    for i, sid in enumerate(all_systems):
        vals = [int(((spoof_dfs[s]["system_id"] == sid) & (spoof_dfs[s]["key"].str.lower() == "spoof")).sum())
                for s in splits]
        ax.bar(splits, vals, bottom=bottom, label=sid, color=_PALETTE[i % len(_PALETTE)], width=0.5, zorder=3)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("spoof clips")
    ax.set_title("Spoof attack systems per split (train/dev vs unseen eval)")
    ax.legend(fontsize=9, frameon=False, ncol=2)
    return fig_to_data_uri(fig)


def build_spoof_summary_table(spoof_dfs: dict[str, pd.DataFrame]) -> str:
    # 스플릿별 요약: 클립 수, bonafide/spoof, 화자 수, 공격 시스템 목록.
    rows = []
    tot_clips = tot_bona = tot_spoof = tot_spk = 0
    for split, df in spoof_dfs.items():
        clips = len(df)
        bona = int((df["key"].str.lower() == "bonafide").sum())
        spoof = clips - bona
        speakers = df["speaker"].nunique()
        systems = df.loc[df["key"].str.lower() == "spoof", "system_id"]
        sys_range = f"{systems.min()}–{systems.max()}" if len(systems) else "—"
        ratio = f"{spoof / max(bona, 1):.1f}:1"
        tot_clips += clips
        tot_bona += bona
        tot_spoof += spoof
        tot_spk += speakers
        rows.append(f"<tr><td>{split}</td><td>{clips:,}</td><td>{bona:,}</td><td>{spoof:,}</td>"
                     f"<td>{ratio}</td><td>{speakers}</td><td>{sys_range}</td></tr>")
    total_row = (f'<tr class="total"><td>Total</td><td>{tot_clips:,}</td><td>{tot_bona:,}</td>'
                 f"<td>{tot_spoof:,}</td><td>{tot_spoof / max(tot_bona, 1):.1f}:1</td>"
                 f"<td>{tot_spk}</td><td></td></tr>")
    note = ('<p class="note">Protocol splits are preserved: train/dev use attacks A01–A06, '
            "eval uses A07–A19 (unseen at train time) — never re-split randomly. "
            "bonafide is kept in full as the real anchor; spoof may be capped per split at "
            "training time (<code>--spoof-cap</code>).</p>")
    return (
        "<table><thead><tr><th>Split</th><th>Clips</th><th>Bonafide (real)</th>"
        "<th>Spoof (fake)</th><th>Spoof:Bona</th><th>Speakers</th><th>Attack systems</th></tr></thead>"
        "<tbody>" + "".join(rows) + total_row + "</tbody></table>" + note
    )


# fname 접두어 -> 소스 코드. 용량을 소스별로 쪼개 길이를 추정할 때 쓴다.
_PREFIX_TO_SOURCE = {"glb_": "GLOBE", "saa_": "SAA"}

# 소스별(코덱별) 대략적인 초당 바이트. curated 오디오는 GLOBE=FLAC@24kHz,
# SAA=mp3 라서 코덱이 다르다. manifest 에는 길이(duration) 컬럼이 없으므로,
# 실제 오디오를 내려받지 않고 "파일 용량 ÷ 초당바이트"로 총 길이를 어림한다.
# 이 값들은 추정치이며(압축률·비트레이트에 따라 달라짐) 화면에도 "est."로 표기한다.
_EST_BYTES_PER_SEC = {"GLOBE": 26_000.0, "SAA": 16_000.0, "other": 20_000.0}


def _source_of(fname: str) -> str:
    for pre, src in _PREFIX_TO_SOURCE.items():
        if fname.startswith(pre):
            return src
    return "other"


def _est_seconds(by_source: dict[str, int]) -> float:
    # 소스별 용량을 각 코덱의 초당바이트로 나눠 더한 "추정" 총 길이(초).
    return sum(b / _EST_BYTES_PER_SEC.get(src, _EST_BYTES_PER_SEC["other"])
               for src, b in by_source.items())


def collect_audio_stats(root: str, classes: list[str]) -> dict[str, dict]:
    """Per-class audio storage stats from object metadata — no audio download.

    For each class we sum ``curated/<CC>/audio/*`` object sizes (``blob.size``
    on GCS, ``stat().st_size`` locally) and split the total by source (glb_/saa_
    filename prefix). Only metadata is read, so this stays cheap even for the
    full pool. Returns ``{cc: {"n": int, "bytes": int, "by_source": {src: bytes}}}``;
    classes whose audio dir can't be listed are simply omitted (size is an
    enhancement — never let it break the counts view).
    """
    # 클래스별 오디오 "용량" 통계를 오브젝트 메타데이터만으로 집계한다(오디오 자체는
    # 내려받지 않음 → curated 는 Standard 스토리지라 비용 부담 없음). blob.size 를
    # 합산하고 fname 접두어(glb_/saa_)로 소스별로 쪼갠다. 나열 실패한 클래스는 조용히
    # 건너뛴다(용량은 부가 정보이므로 클립수 화면을 절대 깨뜨리지 않는다).
    stats: dict[str, dict] = {}
    root = root.rstrip("/")
    for cc in classes:
        try:
            by_source: dict[str, int] = {}
            n = 0
            if root.startswith("gs://"):
                bucket_name, _, prefix = root[len("gs://"):].partition("/")
                audio_prefix = (f"{prefix}/{cc}/audio/" if prefix else f"{cc}/audio/")
                for blob in _gcs().list_blobs(bucket_name, prefix=audio_prefix):
                    if blob.name.endswith("/"):
                        continue
                    base = blob.name.rsplit("/", 1)[-1]
                    src = _source_of(base)
                    by_source[src] = by_source.get(src, 0) + int(blob.size or 0)
                    n += 1
            else:
                adir = Path(root) / cc / "audio"
                if adir.is_dir():
                    for p in adir.iterdir():
                        if not p.is_file():
                            continue
                        src = _source_of(p.name)
                        by_source[src] = by_source.get(src, 0) + p.stat().st_size
                        n += 1
            if n:
                total = sum(by_source.values())
                stats[cc] = {"n": n, "bytes": total, "by_source": by_source,
                             "est_seconds": _est_seconds(by_source)}
                print(f"  {cc}: {n} audio files, {human_size(total)}")
        except Exception as exc:  # 용량 집계 실패는 치명적이지 않다 — 건너뛴다.
            print(f"  ! {cc}: audio stat failed: {exc}")
    return stats


def human_size(nbytes: float) -> str:
    # 바이트를 사람이 읽기 좋은 단위(KB/MB/GB)로 변환.
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def human_duration(seconds: float) -> str:
    # 초를 "Xh Ym" / "Ym Zs" 형태로 변환.
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def fig_to_data_uri(fig: plt.Figure) -> str:
    # matplotlib Figure를 PNG로 렌더링한 뒤 base64로 인코딩해 data URI로 변환.
    # 이렇게 하면 별도 이미지 파일 없이 HTML 하나에 모든 차트를 임베드할 수 있다.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140, transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _bar_labels(ax, xs, vals, fmt="{:.0f}"):
    for x, v in zip(xs, vals):
        if v:
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=13, fontweight="bold", color=_INK)


def chart_clips_per_class(dfs: dict[str, pd.DataFrame]) -> str:
    # 클래스(억양)별 클립 개수를 막대그래프로 표시.
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    classes = list(dfs.keys())
    counts = [len(dfs[c]) for c in classes]
    ax.bar(classes, counts, color=_ACCENT, width=0.6, zorder=3)
    _bar_labels(ax, range(len(classes)), counts)
    ax.set_ylabel("clips")
    ax.set_title("Clips per class")
    return fig_to_data_uri(fig)


def chart_speakers_per_class(dfs: dict[str, pd.DataFrame]) -> str:
    # 클래스별 고유 화자(speaker) 수를 막대그래프로 표시.
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    classes = list(dfs.keys())
    counts = [dfs[c]["speaker"].nunique() for c in classes]
    ax.bar(classes, counts, color=_PALETTE[4], width=0.6, zorder=3)
    _bar_labels(ax, range(len(classes)), counts)
    ax.set_ylabel("unique speakers")
    ax.set_title("Speakers per class")
    return fig_to_data_uri(fig)


def chart_source_breakdown(dfs: dict[str, pd.DataFrame]) -> str:
    # 클래스별로 데이터 출처(source, 예: Common Voice/자체수집 등) 비중을
    # 누적 막대그래프(stacked bar)로 표시.
    all_sources = sorted({s for df in dfs.values() for s in df["source"].unique()})
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    classes = list(dfs.keys())
    bottom = [0] * len(classes)
    for i, source in enumerate(all_sources):
        vals = [int((dfs[c]["source"] == source).sum()) for c in classes]
        ax.bar(classes, vals, bottom=bottom, label=source, color=_PALETTE[i % len(_PALETTE)],
               width=0.6, zorder=3)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("clips")
    ax.set_title("Clips per class by source")
    ax.legend(fontsize=10, frameon=False)
    return fig_to_data_uri(fig)


def chart_storage_per_class(dfs: dict[str, pd.DataFrame], stats: dict[str, dict]) -> str:
    # 클래스별 오디오 총 용량(MB)을 소스별 누적 막대그래프로 표시.
    classes = [c for c in dfs if c in stats]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    if not classes:
        ax.text(0.5, 0.5, "no size metadata", ha="center", va="center", color=_MUTED)
        ax.axis("off")
        return fig_to_data_uri(fig)
    all_sources = sorted({s for c in classes for s in stats[c]["by_source"]})
    bottom = [0.0] * len(classes)
    for i, src in enumerate(all_sources):
        vals = [stats[c]["by_source"].get(src, 0) / (1024 * 1024) for c in classes]
        ax.bar(classes, vals, bottom=bottom, label=src, color=_PALETTE[i % len(_PALETTE)],
               width=0.6, zorder=3)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, c in enumerate(classes):
        ax.text(i, bottom[i], human_size(stats[c]["bytes"]), ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=_INK)
    ax.set_ylabel("audio size (MB)")
    ax.set_title("Storage per class by source")
    ax.legend(fontsize=10, frameon=False)
    return fig_to_data_uri(fig)


def chart_gender_balance(dfs: dict[str, pd.DataFrame]) -> str:
    # 클래스별 성별(F/M/미상 U) 분포를 누적 막대그래프로 표시.
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    classes = list(dfs.keys())
    genders = ["F", "M", "U"]
    colors = {"F": _PALETTE[3], "M": _PALETTE[0], "U": "#cbd5e1"}
    bottom = [0] * len(classes)
    for g in genders:
        vals = [int((dfs[c]["gender"] == g).sum()) for c in classes]
        if not any(vals):
            continue
        ax.bar(classes, vals, bottom=bottom, label=g, color=colors[g], width=0.6, zorder=3)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("clips")
    ax.set_title("Gender balance per class")
    ax.legend(frameon=False)
    return fig_to_data_uri(fig)


def build_headline(dfs: dict[str, pd.DataFrame], stats: dict[str, dict]) -> str:
    # 데이터셋 전체 규모를 한눈에 보는 상단 통계 카드 묶음.
    total_clips = sum(len(df) for df in dfs.values())
    total_speakers = sum(df["speaker"].nunique() for df in dfs.values())
    total_bytes = sum(s["bytes"] for s in stats.values())
    total_secs = sum(s["est_seconds"] for s in stats.values())
    avg_kb = (total_bytes / max(sum(s["n"] for s in stats.values()), 1) / 1024)
    cards = [
        ("🏷️", "Classes", str(len(dfs))),
        ("🎧", "Clips", f"{total_clips:,}"),
        ("🗣️", "Speakers", f"{total_speakers:,}"),
    ]
    if stats:
        cards += [
            ("💾", "Total size", human_size(total_bytes)),
            ("📦", "Avg clip", f"{avg_kb:.0f} KB"),
            ("⏱️", "Est. duration", "≈ " + human_duration(total_secs)),
        ]
    return ('<div class="cards">'
            + "".join(f'<div class="stat"><span class="stat-icon">{icon}</span>'
                      f'<div class="stat-v">{v}</div>'
                      f'<div class="stat-k">{k}</div></div>' for icon, k, v in cards)
            + "</div>")


def build_summary_table(dfs: dict[str, pd.DataFrame], stats: dict[str, dict]) -> str:
    # 클래스별 요약 통계(클립 수, 화자 수, 성비, 용량, 추정 길이, 출처)를 HTML 표로 생성.
    has_size = bool(stats)
    size_head = "<th>Size</th><th>Avg clip</th><th>Est. dur.</th>" if has_size else ""
    rows = []
    tot_clips = tot_spk = tot_bytes = tot_n = 0
    tot_secs = 0.0
    for cc, df in dfs.items():
        clips = len(df)
        speakers = df["speaker"].nunique()
        f = int((df["gender"] == "F").sum())
        m = int((df["gender"] == "M").sum())
        sources = ", ".join(f"{s}={n}" for s, n in df["source"].value_counts().items())
        tot_clips += clips
        tot_spk += speakers
        size_cells = ""
        if has_size:
            st = stats.get(cc)
            if st:
                avg_kb = st["bytes"] / max(st["n"], 1) / 1024
                size_cells = (f"<td>{human_size(st['bytes'])}</td>"
                              f"<td>{avg_kb:.0f} KB</td>"
                              f"<td>≈ {human_duration(st['est_seconds'])}</td>")
                tot_bytes += st["bytes"]
                tot_n += st["n"]
                tot_secs += st["est_seconds"]
            else:
                size_cells = "<td>—</td><td>—</td><td>—</td>"
        rows.append(f"<tr><td>{cc}</td><td>{clips}</td><td>{speakers}</td>"
                     f"<td>{f} / {m}</td>{size_cells}<td>{sources}</td></tr>")
    # 합계 행
    tot_size_cells = ""
    if has_size:
        avg_kb = tot_bytes / max(tot_n, 1) / 1024
        tot_size_cells = (f"<td>{human_size(tot_bytes)}</td><td>{avg_kb:.0f} KB</td>"
                          f"<td>≈ {human_duration(tot_secs)}</td>")
    total_row = (f'<tr class="total"><td>Total</td><td>{tot_clips}</td><td>{tot_spk}</td>'
                 f"<td></td>{tot_size_cells}<td></td></tr>")
    note = ('<p class="note">Size is exact (object metadata). '
            "“Est. dur.” is estimated from file size per codec (GLOBE FLAC / SAA mp3) — "
            "manifests have no per-clip duration, so treat it as a rough total.</p>"
            if has_size else "")
    return (
        "<table><thead><tr><th>Class</th><th>Clips</th><th>Speakers</th>"
        f"<th>F / M</th>{size_head}<th>Sources</th></tr></thead><tbody>"
        + "".join(rows) + total_row + "</tbody></table>" + note
    )


# 리포트 "본문" 템플릿(표 + 차트). 정적 HTML 파일에도, 서버 모드의 새로고침
# 응답(innerHTML 교체)에도 그대로 재사용된다.
BODY_TEMPLATE = """<p class="meta">Source root: <code>{root}</code> &middot; generated {generated}</p>
{headline}
<div class="card table-card">{table}</div>
<div class="charts">
<div class="chart-card"><img src="{c1}" alt="Clips per class"></div>
<div class="chart-card"><img src="{c2}" alt="Speakers per class"></div>
<div class="chart-card"><img src="{c3}" alt="Clips per class by source"></div>
<div class="chart-card"><img src="{c4}" alt="Gender balance per class"></div>
<div class="chart-card"><img src="{c5}" alt="Storage per class by source"></div>
</div>
{spoof_section}
"""

# fake(합성음성) 탐지용 spoof 코퍼스 섹션. spoof_dfs가 없으면(root 미지정/미발견)
# 조용히 빈 문자열을 반환해 country-only 리포트도 그대로 동작한다.
SPOOF_SECTION_TEMPLATE = """<h2 class="section-title">Spoof (fake-voice) corpus &middot; ASVspoof 2019 LA</h2>
<p class="meta">Spoof root: <code>{spoof_root}</code></p>
<div class="card table-card">{table}</div>
<div class="charts">
<div class="chart-card"><img src="{c1}" alt="Spoof composition per split"></div>
<div class="chart-card"><img src="{c2}" alt="Spoof attack systems per split"></div>
</div>
"""

# 공통 디자인 토큰/베이스 스타일. serve_dataset_report.py도 이 팔레트를 그대로 쓴다
# (정적 리포트와 라이브 대시보드가 같은 룩앤필을 갖도록).
SHARED_STYLE = """
:root {{
  --bg: #f1f5f9; --bg-grad: linear-gradient(180deg, #eef2ff 0%, #f1f5f9 320px);
  --surface: #ffffff; --border: #e2e8f0; --ink: #0f172a; --muted: #64748b;
  --accent: #6366f1; --accent-2: #8b5cf6; --shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -12px rgba(15,23,42,.12);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #0b1120; --bg-grad: linear-gradient(180deg, #151d34 0%, #0b1120 320px);
    --surface: #131b2e; --border: #253046; --ink: #e6e9f2; --muted: #93a1bd;
    --accent: #818cf8; --accent-2: #a78bfa; --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px -12px rgba(0,0,0,.5);
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  font-family: "Segoe UI", Inter, system-ui, sans-serif; margin: 0; padding: 2.2rem clamp(1rem, 4vw, 3rem) 4rem;
  color: var(--ink); background: var(--bg-grad), var(--bg); background-attachment: fixed;
  min-height: 100vh; line-height: 1.5;
}}
h1 {{ margin: 0; font-size: 1.6rem; font-weight: 800; letter-spacing: -.01em; }}
.subtitle {{ margin: .2rem 0 0; color: var(--muted); font-size: .92rem; }}
.meta {{ color: var(--muted); margin: .2rem 0 1.4rem; font-size: .85rem; }}
.meta code {{ background: var(--surface); border: 1px solid var(--border); border-radius: 5px; padding: .05rem .4rem; }}
.card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; box-shadow: var(--shadow); }}
.table-card {{ padding: .4rem 1.2rem; overflow-x: auto; margin-bottom: 1.6rem; }}
table {{ border-collapse: collapse; width: 100%; margin: .6rem 0; font-size: .9rem; }}
th, td {{ padding: .55rem .9rem; text-align: left; }}
th {{ color: var(--muted); font-size: .74rem; text-transform: uppercase; letter-spacing: .04em;
     border-bottom: 1px solid var(--border); font-weight: 700; }}
tbody tr:not(.total) {{ border-bottom: 1px solid var(--border); }}
tbody tr:not(.total):hover {{ background: color-mix(in srgb, var(--accent) 6%, transparent); }}
tr.total td {{ font-weight: 700; border-top: 2px solid var(--accent); background: color-mix(in srgb, var(--accent) 7%, transparent); }}
.note {{ color: var(--muted); font-size: .82rem; max-width: 680px; margin: .8rem 0 1rem; }}
.cards {{ display: flex; flex-wrap: wrap; gap: .9rem; margin: 1.3rem 0 1.8rem; }}
.stat {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; box-shadow: var(--shadow);
        padding: .9rem 1.3rem; min-width: 118px; position: relative; overflow: hidden; }}
.stat::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 3px;
                background: linear-gradient(180deg, var(--accent), var(--accent-2)); }}
.stat-icon {{ font-size: 1.1rem; opacity: .85; }}
.stat-v {{ font-size: 1.5rem; font-weight: 800; color: var(--ink); margin-top: .15rem; }}
.stat-k {{ font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; margin-top: .1rem; }}
.section-title {{ margin: 2.2rem 0 .9rem; font-size: 1.15rem; font-weight: 800; }}
.charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 1.4rem; }}
.chart-card {{ background: #ffffff; border: 1px solid var(--border); border-radius: 14px;
              box-shadow: var(--shadow); padding: 1.4rem 1.5rem; display: flex; align-items: center; justify-content: center; }}
.chart-card img {{ max-width: 100%; width: 100%; height: auto; display: block; }}
@media (max-width: 600px) {{
  .charts {{ grid-template-columns: 1fr; }}
}}
"""

# 정적 파일용 페이지 전체 템플릿(<html>/<head> 포함). 서버 모드는 자체 페이지에
# render_body()의 결과만 끼워 넣으므로 이 템플릿을 쓰지 않는다.
PAGE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Curated dataset report</title>
<style>""" + SHARED_STYLE + """
</style></head>
<body>
<h1>Curated dataset report</h1>
<p class="subtitle">Speech-accent corpus &middot; class balance, sources &amp; storage</p>
{body}
</body></html>
"""


def render_spoof_section(spoof_root: str | None, spoof_splits: list[str]) -> str:
    # spoof_root가 없으면(비활성) 빈 문자열 — country-only 리포트 하위호환.
    if not spoof_root:
        return ""
    spoof_dfs = collect_spoof_manifests(spoof_root, spoof_splits)
    if not spoof_dfs:
        return ""
    return SPOOF_SECTION_TEMPLATE.format(
        spoof_root=spoof_root,
        table=build_spoof_summary_table(spoof_dfs),
        c1=chart_spoof_composition(spoof_dfs),
        c2=chart_spoof_attack_systems(spoof_dfs),
    )


def render_body(root: str, dfs: dict[str, pd.DataFrame], spoof_root: str | None = None,
                 spoof_splits: list[str] = SPOOF_SPLITS) -> str:
    """Table + charts only — no <html>/<head> wrapper. Shared by CLI and server.

    Also reads per-class audio storage stats (object metadata only, no audio
    download) so the report shows total/avg file size and an estimated total
    duration alongside the clip counts. If ``spoof_root`` is given, appends a
    second section for the fake-voice (ASVspoof) corpus — a separate label
    axis/manifest schema from the country classes (DATASET.md §10).
    """
    stats = collect_audio_stats(root, list(dfs.keys()))
    return BODY_TEMPLATE.format(
        root=root,
        generated=datetime.datetime.now().isoformat(timespec="seconds"),
        headline=build_headline(dfs, stats),
        table=build_summary_table(dfs, stats),
        c1=chart_clips_per_class(dfs),
        c2=chart_speakers_per_class(dfs),
        c3=chart_source_breakdown(dfs),
        c4=chart_gender_balance(dfs),
        c5=chart_storage_per_class(dfs, stats),
        spoof_section=render_spoof_section(spoof_root, spoof_splits),
    )


def build_report_html(root: str, dfs: dict[str, pd.DataFrame], spoof_root: str | None = None,
                       spoof_splits: list[str] = SPOOF_SPLITS) -> str:
    """Full standalone HTML page — used for the static file output."""
    return PAGE_TEMPLATE.format(body=render_body(root, dfs, spoof_root, spoof_splits))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="curated/ root (gs:// URI or local dir)")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    ap.add_argument("--spoof-root", default=DEFAULT_SPOOF_ROOT,
                     help="curated_spoof/asvspoof2019_la/ root; pass '' to disable the spoof section")
    ap.add_argument("--spoof-splits", nargs="+", default=SPOOF_SPLITS)
    ap.add_argument("--out", type=Path, default=REPORT_OUT)
    args = ap.parse_args()

    dfs = collect_manifests(args.root, args.classes)
    if not dfs:
        # 읽어들인 매니페스트가 하나도 없으면 리포트를 만들 수 없으므로 즉시 중단.
        raise SystemExit("No manifests found — check --root and --classes.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        build_report_html(args.root, dfs, args.spoof_root or None, args.spoof_splits),
        encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
