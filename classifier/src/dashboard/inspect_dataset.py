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

DEFAULT_ROOT = "gs://qi-ucsd-speech-us/curated"
DEFAULT_CLASSES = ["US", "UK", "IN", "NG", "CA", "JP", "CN", "AU", "KR"]
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
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chart_clips_per_class(dfs: dict[str, pd.DataFrame]) -> str:
    # 클래스(억양)별 클립 개수를 막대그래프로 표시.
    fig, ax = plt.subplots(figsize=(6, 4))
    classes = list(dfs.keys())
    counts = [len(dfs[c]) for c in classes]
    ax.bar(classes, counts, color="#4C78A8")
    for i, v in enumerate(counts):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("clips")
    ax.set_title("Clips per class")
    return fig_to_data_uri(fig)


def chart_speakers_per_class(dfs: dict[str, pd.DataFrame]) -> str:
    # 클래스별 고유 화자(speaker) 수를 막대그래프로 표시.
    fig, ax = plt.subplots(figsize=(6, 4))
    classes = list(dfs.keys())
    counts = [dfs[c]["speaker"].nunique() for c in classes]
    ax.bar(classes, counts, color="#72B7B2")
    for i, v in enumerate(counts):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("unique speakers")
    ax.set_title("Speakers per class")
    return fig_to_data_uri(fig)


def chart_source_breakdown(dfs: dict[str, pd.DataFrame]) -> str:
    # 클래스별로 데이터 출처(source, 예: Common Voice/자체수집 등) 비중을
    # 누적 막대그래프(stacked bar)로 표시.
    all_sources = sorted({s for df in dfs.values() for s in df["source"].unique()})
    fig, ax = plt.subplots(figsize=(7, 4))
    classes = list(dfs.keys())
    bottom = [0] * len(classes)
    colors = plt.get_cmap("tab10").colors
    for i, source in enumerate(all_sources):
        vals = [int((dfs[c]["source"] == source).sum()) for c in classes]
        ax.bar(classes, vals, bottom=bottom, label=source, color=colors[i % len(colors)])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("clips")
    ax.set_title("Clips per class by source")
    ax.legend(fontsize=8)
    return fig_to_data_uri(fig)


def chart_storage_per_class(dfs: dict[str, pd.DataFrame], stats: dict[str, dict]) -> str:
    # 클래스별 오디오 총 용량(MB)을 소스별 누적 막대그래프로 표시.
    classes = [c for c in dfs if c in stats]
    fig, ax = plt.subplots(figsize=(6, 4))
    if not classes:
        ax.text(0.5, 0.5, "no size metadata", ha="center", va="center")
        ax.axis("off")
        return fig_to_data_uri(fig)
    all_sources = sorted({s for c in classes for s in stats[c]["by_source"]})
    colors = plt.get_cmap("tab10").colors
    bottom = [0.0] * len(classes)
    for i, src in enumerate(all_sources):
        vals = [stats[c]["by_source"].get(src, 0) / (1024 * 1024) for c in classes]
        ax.bar(classes, vals, bottom=bottom, label=src, color=colors[i % len(colors)])
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, c in enumerate(classes):
        ax.text(i, bottom[i], human_size(stats[c]["bytes"]), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("audio size (MB)")
    ax.set_title("Storage per class by source")
    ax.legend(fontsize=8)
    return fig_to_data_uri(fig)


def chart_gender_balance(dfs: dict[str, pd.DataFrame]) -> str:
    # 클래스별 성별(F/M/미상 U) 분포를 누적 막대그래프로 표시.
    fig, ax = plt.subplots(figsize=(7, 4))
    classes = list(dfs.keys())
    genders = ["F", "M", "U"]
    colors = {"F": "#E45756", "M": "#4C78A8", "U": "#B0B0B0"}
    bottom = [0] * len(classes)
    for g in genders:
        vals = [int((dfs[c]["gender"] == g).sum()) for c in classes]
        if not any(vals):
            continue
        ax.bar(classes, vals, bottom=bottom, label=g, color=colors[g])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("clips")
    ax.set_title("Gender balance per class")
    ax.legend()
    return fig_to_data_uri(fig)


def build_headline(dfs: dict[str, pd.DataFrame], stats: dict[str, dict]) -> str:
    # 데이터셋 전체 규모를 한눈에 보는 상단 통계 카드 묶음.
    total_clips = sum(len(df) for df in dfs.values())
    total_speakers = sum(df["speaker"].nunique() for df in dfs.values())
    total_bytes = sum(s["bytes"] for s in stats.values())
    total_secs = sum(s["est_seconds"] for s in stats.values())
    avg_kb = (total_bytes / max(sum(s["n"] for s in stats.values()), 1) / 1024)
    cards = [
        ("Classes", str(len(dfs))),
        ("Clips", f"{total_clips:,}"),
        ("Speakers", f"{total_speakers:,}"),
    ]
    if stats:
        cards += [
            ("Total size", human_size(total_bytes)),
            ("Avg clip", f"{avg_kb:.0f} KB"),
            ("Est. duration", "≈ " + human_duration(total_secs)),
        ]
    return ('<div class="cards">'
            + "".join(f'<div class="stat"><div class="stat-v">{v}</div>'
                      f'<div class="stat-k">{k}</div></div>' for k, v in cards)
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
{table}
<div class="charts">
<img src="{c1}">
<img src="{c2}">
<img src="{c3}">
<img src="{c4}">
<img src="{c5}">
</div>
"""

# 정적 파일용 페이지 전체 템플릿(<html>/<head> 포함). 서버 모드는 자체 페이지에
# render_body()의 결과만 끼워 넣으므로 이 템플릿을 쓰지 않는다.
PAGE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Dataset report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
h1 {{ margin-bottom: 0; }}
.meta {{ color: #666; margin-top: 0.2rem; }}
table {{ border-collapse: collapse; margin: 1.5rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }}
th {{ background: #f2f2f2; }}
tr.total td {{ font-weight: 700; background: #fafafa; border-top: 2px solid #999; }}
.note {{ color: #888; font-size: 0.85rem; max-width: 640px; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 0.8rem; margin: 1.2rem 0; }}
.stat {{ background: #f7f9fc; border: 1px solid #e2e8f0; border-radius: 8px;
         padding: 0.7rem 1.1rem; min-width: 92px; }}
.stat-v {{ font-size: 1.35rem; font-weight: 700; color: #1e293b; }}
.stat-k {{ font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: .03em; }}
.charts {{ display: flex; flex-wrap: wrap; gap: 1.5rem; }}
.charts img {{ max-width: 100%; border: 1px solid #ddd; }}
</style></head>
<body>
<h1>Curated dataset report</h1>
{body}
</body></html>
"""


def render_body(root: str, dfs: dict[str, pd.DataFrame]) -> str:
    """Table + charts only — no <html>/<head> wrapper. Shared by CLI and server.

    Also reads per-class audio storage stats (object metadata only, no audio
    download) so the report shows total/avg file size and an estimated total
    duration alongside the clip counts.
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
    )


def build_report_html(root: str, dfs: dict[str, pd.DataFrame]) -> str:
    """Full standalone HTML page — used for the static file output."""
    return PAGE_TEMPLATE.format(body=render_body(root, dfs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="curated/ root (gs:// URI or local dir)")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    ap.add_argument("--out", type=Path, default=REPORT_OUT)
    args = ap.parse_args()

    dfs = collect_manifests(args.root, args.classes)
    if not dfs:
        # 읽어들인 매니페스트가 하나도 없으면 리포트를 만들 수 없으므로 즉시 중단.
        raise SystemExit("No manifests found — check --root and --classes.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_report_html(args.root, dfs), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
