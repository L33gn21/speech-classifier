"""Live model-tester dashboard — pick a trained model on GCS and run it on
audio you upload or record in the browser. No model is ever downloaded to the
user's machine: the Cloud Run instance pulls the selected model's weights from
GCS into its own memory and runs inference server-side.

Sibling to serve_dataset_report.py, but deliberately a *separate* Cloud Run
service: this one needs torch + transformers + ffmpeg (a multi-GB image), so
folding it into the lightweight dataset dashboard would slow that page's cold
starts. The two pages just link to each other.

Config is read from env vars so the same module works both as a local script
(`python serve_model_tester.py`) and as a gunicorn app on Cloud Run
(`gunicorn serve_model_tester:app`):

    MODEL_ROOT        gs:// prefix holding <JOB_NAME>/model/ dirs
                      (default: gs://qi-ucsd-speech-us/outputs/classifier)
    DASHBOARD_USER    login user (shared with the dataset dashboard)
    DASHBOARD_PASS    login password
    DATASET_DASHBOARD_URL  optional link back to the dataset dashboard
    API_KEY           static key other servers send via X-API-Key to hit /api/*
    PORT              listen port (Cloud Run injects this; default 8766)

/api/* is a separate, machine-facing surface (X-API-Key header auth instead of
the browser session login) — see api_key_required and the /api/models,
/api/metrics/<job>, /api/predict routes near the bottom of this file.

Usage:
    python serve_model_tester.py
    # then open http://127.0.0.1:8766/
"""
# 학습된 모델을 GCS에서 골라, 브라우저에서 업로드하거나 즉석 녹음한 오디오로
# 바로 테스트해보는 라이브 대시보드.
#
# 핵심: 모델은 사용자 PC로 절대 내려오지 않는다. Cloud Run 인스턴스가 선택된
# 모델의 가중치를 GCS에서 자기 메모리로 받아 서버 측에서 추론을 돌린다.
# 사용자는 오디오를 올리거나 녹음만 한다.
#
# serve_dataset_report.py 와 형제지간이지만, 일부러 "별도의" Cloud Run 서비스로
# 둔다. 이쪽은 torch + transformers + ffmpeg 가 필요해 이미지가 수 GB로 무겁기
# 때문에, 가벼운 데이터셋 대시보드에 합치면 그 페이지 콜드스타트까지 느려진다.
# 두 페이지는 서로 링크로만 연결한다.
from __future__ import annotations

import json
import os
import subprocess
import threading
from functools import wraps
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, redirect, request, session, url_for
from google.cloud import storage
from transformers import AutoFeatureExtractor

from config import FAKE_LABELS, LABELS, SAMPLE_RATE
from model import AccentClassifier, load_from_dir

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(32)

# 단일 사용자 로그인 정보. 데이터셋 대시보드와 동일한 기본값을 공유한다.
DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "geonah")
DASHBOARD_PASS = os.environ.get("DASHBOARD_PASS", "dmdlsldk2!")

# 다른 서버가 /api/* 를 호출할 때 쓰는 정적 API 키. 데모용 — 회전/스코프 없이
# 헤더 값만 비교한다. 배포 시 deploy.sh 가 env var로 주입한다.
API_KEY = os.environ.get("API_KEY", "dev-key-change-me")

# 학습 job들이 쌓이는 GCS 접두어. 각 job은 <MODEL_ROOT>/<JOB_NAME>/model/ 아래에
# model.safetensors / label_config.json / preprocessor_config.json / final_metrics.json 을 갖는다.
MODEL_ROOT = os.environ.get(
    "MODEL_ROOT", "gs://qi-ucsd-speech-us/outputs/classifier"
).rstrip("/")

# 데이터셋 대시보드로 돌아가는 링크(있으면 상단에 표시).
DATASET_DASHBOARD_URL = os.environ.get("DATASET_DASHBOARD_URL", "")

# 최대 업로드 크기(25MB). 짧은 테스트 클립이면 충분하고, 과도한 업로드를 막는다.
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# ---------------------------------------------------------------------------
# GCS helpers  —  GCS 도우미
# ---------------------------------------------------------------------------
def _split_gs(uri: str) -> tuple[str, str]:
    # "gs://bucket/prefix" -> ("bucket", "prefix")
    assert uri.startswith("gs://"), uri
    rest = uri[len("gs://"):]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix


_gcs_lock = threading.Lock()
_gcs_client: storage.Client | None = None


def _client() -> storage.Client:
    # storage.Client는 스레드마다 새로 만들 필요 없이 하나를 재사용한다.
    global _gcs_client
    with _gcs_lock:
        if _gcs_client is None:
            _gcs_client = storage.Client()
        return _gcs_client


def list_models() -> list[dict]:
    """List trained models under MODEL_ROOT, newest first.

    A model = a job dir that contains ``model/model.safetensors``. We attach
    each model's held-out metrics (from final_metrics.json) so the dropdown can
    show accuracy, and sort by job name descending (names are timestamped, so
    lexicographic == chronological).
    """
    # MODEL_ROOT 아래에서 model/model.safetensors 를 가진 job 폴더들을 모델로 보고,
    # 각 모델의 final_metrics.json(정확도 등)을 함께 붙여 최신순으로 돌려준다.
    bucket_name, prefix = _split_gs(MODEL_ROOT)
    prefix = prefix.rstrip("/") + "/"
    bucket = _client().bucket(bucket_name)

    # 접두어 바로 아래의 "폴더"(job 이름)만 얻기 위해 delimiter로 훑는다.
    it = _client().list_blobs(bucket_name, prefix=prefix, delimiter="/")
    list(it)  # prefixes는 순회를 마쳐야 채워진다
    job_prefixes = sorted(it.prefixes, reverse=True)

    models: list[dict] = []
    for jp in job_prefixes:
        job = jp[len(prefix):].strip("/")
        weights = bucket.blob(f"{prefix}{job}/model/model.safetensors")
        if not weights.exists():
            continue  # 학습이 끝나 저장까지 마친 job만 노출
        entry: dict = {"job": job}
        metrics_blob = bucket.blob(f"{prefix}{job}/model/final_metrics.json")
        if metrics_blob.exists():
            try:
                m = json.loads(metrics_blob.download_as_bytes())
                # 멀티태스크 잡은 "test_accuracy" 대신 "test_country_accuracy"/
                # "test_fake_macro_f1" 키를 쓴다 — 둘 다 폴백으로 받아준다.
                multitask = bool(m.get("train_config", {}).get("multitask")) or "test_fake_macro_f1" in m
                entry["multitask"] = multitask
                entry["test_accuracy"] = m.get("test_accuracy", m.get("test_country_accuracy"))
                entry["eval_accuracy"] = m.get("eval_accuracy", m.get("eval_country_accuracy"))
                entry["macro_f1"] = m.get("test_macro_f1", m.get("test_country_macro_f1"))
                entry["fake_macro_f1"] = m.get("test_fake_macro_f1")
                # 나라별 상세치를 볼 수 있는 모델인지 표시(드롭다운은 가볍게 유지하고,
                # 상세 지표 자체는 선택 시 /metrics/<job> 로 따로 받는다).
                entry["has_detail"] = ("test_detail" in m or "eval_detail" in m
                                       or any(k.startswith("test_f1_") for k in m))
            except Exception:
                pass
        models.append(entry)
    return models


def get_metrics(job: str) -> dict:
    """Full final_metrics.json for one job, normalized for the tester frontend.

    Returns the raw metrics plus a ``detail`` block (labels + per-class
    precision/recall/f1/support + confusion matrix) when the model was trained
    with the detailed report. Older models only carry flat ``test_f1_<LABEL>``
    scalars, so we synthesize a per-class F1 view from those as a fallback — the
    dashboard then shows per-country F1 bars even for pre-existing models, just
    without the confusion matrix.
    """
    # 선택된 모델의 final_metrics.json 전체를 프론트가 쓰기 좋은 형태로 돌려준다.
    # 신형 모델은 test_detail(혼동행렬·정밀도·재현율)을 갖고, 구형 모델은 평면적인
    # test_f1_<LABEL> 스칼라만 있으므로 그것으로 클래스별 F1 뷰를 합성한다(혼동행렬은 없음).
    bucket_name, prefix = _split_gs(MODEL_ROOT)
    prefix = prefix.rstrip("/") + "/"
    bucket = _client().bucket(bucket_name)
    blob = bucket.blob(f"{prefix}{job}/model/final_metrics.json")
    if not blob.exists():
        return {"job": job, "metrics": None, "detail": None}
    m = json.loads(blob.download_as_bytes())
    multitask = bool(m.get("train_config", {}).get("multitask")) or "test_fake_macro_f1" in m

    def summary(split: str) -> dict:
        # 멀티태스크 잡은 country 지표가 "{split}_country_*" 키를 쓴다(§list_models).
        return {
            "accuracy": m.get(f"{split}_accuracy", m.get(f"{split}_country_accuracy")),
            "macro_f1": m.get(f"{split}_macro_f1", m.get(f"{split}_country_macro_f1")),
            "loss": m.get(f"{split}_loss"),
            "fake_accuracy": m.get(f"{split}_fake_accuracy"),
            "fake_macro_f1": m.get(f"{split}_fake_macro_f1"),
        }

    # 상세 블록: 신형은 그대로 사용, 구형은 f1 스칼라로 합성.
    detail = m.get("test_detail") or m.get("eval_detail")
    if detail is None:
        for split in ("test", "eval"):
            f1s = {k[len(f"{split}_f1_"):]: v for k, v in m.items()
                   if k.startswith(f"{split}_f1_")}
            if f1s:
                detail = {
                    "labels": list(f1s.keys()),
                    "per_class": {name: {"f1": v} for name, v in f1s.items()},
                    "confusion_matrix": None,
                }
                break

    # fake 헤드 클래스별(real/fake) F1 — 멀티태스크 잡만 갖는다.
    fake_detail = None
    for split in ("test", "eval"):
        f1s = {k[len(f"{split}_fake_f1_"):]: v for k, v in m.items()
               if k.startswith(f"{split}_fake_f1_")}
        if f1s:
            fake_detail = {
                "labels": list(f1s.keys()),
                "per_class": {name: {"f1": v} for name, v in f1s.items()},
            }
            break

    return {
        "job": job,
        "multitask": multitask,
        "test": summary("test"),
        "eval": summary("eval"),
        "detail": detail,
        "fake_detail": fake_detail,
    }


# ---------------------------------------------------------------------------
# Model cache  —  로드된 모델 캐시
# ---------------------------------------------------------------------------
# job 이름 -> (model, feature_extractor, labels). 첫 요청 때만 GCS에서 받아
# 메모리에 올리고(느림), 이후 같은 인스턴스가 살아있는 동안은 재사용(빠름).
_models: dict[str, tuple] = {}
_model_lock = threading.Lock()


def _download_model_dir(job: str, dst: Path) -> None:
    # 선택된 job의 model/ 아래 추론에 필요한 파일들만 컨테이너 임시 디스크로 받는다.
    # (checkpoint-*/ 같은 학습 중간물은 제외 — 추론엔 불필요하고 용량만 크다.)
    bucket_name, prefix = _split_gs(MODEL_ROOT)
    prefix = prefix.rstrip("/") + "/"
    bucket = _client().bucket(bucket_name)
    wanted = [
        "model.safetensors",
        "label_config.json",
        "preprocessor_config.json",
        # model_config.json 이 없으면 load_from_dir 가 항상 레거시 기본 구조(country만,
        # fake_head=False)로 골격을 짓는다 — 멀티태스크/attentive 모델은 이 파일이 필수.
        "model_config.json",
    ]
    dst.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        blob = bucket.blob(f"{prefix}{job}/model/{name}")
        if blob.exists():
            blob.download_to_filename(str(dst / name))


def get_model(job: str):
    # job에 해당하는 모델을 캐시에서 꺼내거나, 없으면 GCS에서 받아 로드해 캐시에 넣는다.
    with _model_lock:
        if job in _models:
            return _models[job]

    model_dir = Path("/tmp/models") / job
    if not (model_dir / "model.safetensors").exists():
        _download_model_dir(job, model_dir)

    labels = LABELS
    fake_labels = FAKE_LABELS
    cfg = model_dir / "label_config.json"
    if cfg.exists():
        cfg_data = json.loads(cfg.read_text())
        labels = cfg_data["labels"]
        fake_labels = cfg_data.get("fake_labels", FAKE_LABELS)

    # 백본은 config로만 짓고(HF 재다운로드 없음) 우리 safetensors로 덮어쓴다.
    # load_from_dir 가 model_config.json 을 읽어 학습 때와 동일한 백본·헤드·fake_head
    # 구조를 만든다(구버전 체크포인트는 레거시 기본값=country만 으로 폴백).
    model = load_from_dir(model_dir, num_labels=len(labels))
    from safetensors.torch import load_file

    state = load_file(str(model_dir / "model.safetensors"))
    model.load_state_dict(state)
    model.eval()

    fe = AutoFeatureExtractor.from_pretrained(model_dir)
    loaded = (model, fe, labels, fake_labels)
    with _model_lock:
        _models[job] = loaded
    return loaded


# ---------------------------------------------------------------------------
# Audio decoding  —  오디오 디코딩
# ---------------------------------------------------------------------------
def decode_audio(raw: bytes) -> np.ndarray:
    """Decode arbitrary audio bytes (wav/mp3/webm/ogg/m4a...) to float32 mono
    16 kHz using ffmpeg. Browser MediaRecorder emits webm/opus, so we lean on
    ffmpeg rather than torchaudio/soundfile to cover every container.
    """
    # 브라우저 녹음은 webm/opus로 오고 업로드는 mp3/wav 등 제각각이라, 컨테이너를
    # 가리지 않는 ffmpeg로 통일해서 16kHz 모노 float32로 디코딩한다.
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
        input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise ValueError(f"ffmpeg failed to decode audio: {proc.stderr.decode(errors='ignore')[:400]}")
    wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if wav.size == 0:
        raise ValueError("decoded audio is empty")
    return wav


@torch.no_grad()
def run_inference(job: str, raw: bytes) -> dict:
    model, fe, labels, fake_labels = get_model(job)
    wav = decode_audio(raw)
    inputs = fe([wav], sampling_rate=SAMPLE_RATE, return_attention_mask=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
    order = np.argsort(probs)[::-1]
    ranked = [{"label": labels[i], "prob": float(probs[i])} for i in order]

    fake_result = None
    if getattr(model, "fake_head_enabled", False) and out.fake_logits is not None:
        fake_probs = torch.softmax(out.fake_logits, dim=-1)[0].cpu().numpy()
        verdict = fake_labels[int(np.argmax(fake_probs))]
        fake_result = {
            "verdict": verdict,
            "probs": [{"label": fake_labels[i], "prob": float(fake_probs[i])}
                      for i in range(len(fake_labels))],
        }

    return {
        "duration_s": round(wav.size / SAMPLE_RATE, 2),
        "predictions": ranked,
        "fake": fake_result,
    }


# ---------------------------------------------------------------------------
# Auth  —  로그인 (데이터셋 대시보드와 동일한 패턴)
# ---------------------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def api_key_required(view):
    # 세션 로그인이 아니라 X-API-Key 헤더로 인증하는 머신용 라우트에 붙인다.
    @wraps(view)
    def wrapped(*args, **kwargs):
        key = request.headers.get("X-API-Key", "")
        if key != API_KEY:
            return jsonify(ok=False, error="invalid or missing X-API-Key header"), 401
        return view(*args, **kwargs)

    return wrapped


# /api/* 를 브라우저에서 직접 호출하는 외부 서비스(해커톤 프론트엔드)를 위한 CORS.
# 세션 로그인 라우트는 same-origin만 쓰므로 대상 밖 — /api/* 에만 헤더를 붙인다.
CORS_ALLOWED_ORIGINS = {
    o.strip()
    for o in os.environ.get(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:8000,https://jinwoong-team-hackertone2026-4nsi.onrender.com",
    ).split(",")
    if o.strip()
}


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if request.path.startswith("/api/") and origin in CORS_ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "X-API-Key, Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/<path:_path>", methods=["OPTIONS"])
def api_cors_preflight(_path):
    # 프리플라이트는 API 키 없이 온다 — 인증 없이 204만 돌려주고 위 after_request가
    # 헤더를 붙인다.
    return "", 204


LOGIN_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Login — Model tester</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 0; height: 100vh; display: flex;
       align-items: center; justify-content: center; background: #f5f5f5; color: #222; }}
form {{ background: #fff; padding: 2rem 2.5rem; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.15);
        display: flex; flex-direction: column; gap: 0.8rem; min-width: 260px; }}
h1 {{ font-size: 1.2rem; margin: 0 0 0.5rem; }}
input {{ padding: 0.5rem 0.6rem; font-size: 1rem; border: 1px solid #ccc; border-radius: 4px; }}
button {{ padding: 0.5rem; font-size: 1rem; cursor: pointer; border: none; border-radius: 4px;
          background: #2563eb; color: #fff; }}
.error {{ color: #c00; margin: 0; }}
</style></head>
<body>
<form method="post" action="/login">
<h1>Model tester login</h1>
{error}
<input type="text" name="username" placeholder="Username" autofocus required>
<input type="password" name="password" placeholder="Password" required>
<button type="submit">Log in</button>
</form>
</body></html>
"""


@app.get("/login")
def login():
    return LOGIN_PAGE.format(error="")


@app.post("/login")
def login_submit():
    if request.form.get("username") == DASHBOARD_USER and request.form.get("password") == DASHBOARD_PASS:
        session["logged_in"] = True
        return redirect(url_for("index"))
    return LOGIN_PAGE.format(error='<p class="error">Invalid username or password.</p>'), 401


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Page  —  메인 페이지
# ---------------------------------------------------------------------------
PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Model tester</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; max-width: 760px; }}
h1 {{ margin: 0 0 0.2rem; }}
.topnav {{ color: #666; margin-bottom: 1.5rem; }}
.topnav a {{ margin-right: 1rem; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1.2rem 1.4rem; margin-bottom: 1.2rem; }}
label {{ font-weight: 600; display: block; margin-bottom: 0.4rem; }}
select, input[type=file] {{ font-size: 1rem; padding: 0.4rem; width: 100%; box-sizing: border-box; }}
button {{ font-size: 1rem; padding: 0.5rem 1rem; cursor: pointer; border: none; border-radius: 4px;
          background: #2563eb; color: #fff; }}
button:disabled {{ background: #9db8f0; cursor: default; }}
button.secondary {{ background: #6b7280; }}
.row {{ display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }}
#status {{ color: #666; margin-left: 0.5rem; }}
.error {{ color: #c00; }}
.bar-wrap {{ margin: 0.35rem 0; }}
.bar-label {{ display: flex; justify-content: space-between; font-variant-numeric: tabular-nums; }}
.bar-track {{ background: #eee; border-radius: 4px; height: 14px; overflow: hidden; }}
.bar-fill {{ background: #2563eb; height: 100%; }}
.bar-wrap.top .bar-fill {{ background: #16a34a; }}
small {{ color: #888; }}
audio {{ width: 100%; margin-top: 0.6rem; }}
/* --- model performance viz --- */
.cards {{ display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.4rem 0 1rem; }}
.stat {{ background: #f7f9fc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.6rem 0.9rem; min-width: 80px; }}
.stat-v {{ font-size: 1.3rem; font-weight: 700; color: #1e293b; font-variant-numeric: tabular-nums; }}
.stat-k {{ font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: .03em; }}
.sec-title {{ font-weight: 600; margin: 1rem 0 0.4rem; }}
.pc-row {{ display: grid; grid-template-columns: 2.5rem 1fr 3.2rem; align-items: center; gap: 0.5rem; margin: 0.28rem 0; }}
.pc-name {{ font-weight: 600; }}
.pc-track {{ background: #eee; border-radius: 4px; height: 16px; overflow: hidden; }}
.pc-fill {{ height: 100%; background: #2563eb; }}
.pc-val {{ text-align: right; font-variant-numeric: tabular-nums; color: #334155; }}
.pc-sub {{ color: #94a3b8; font-size: 0.75rem; }}
table.cm {{ border-collapse: collapse; margin-top: 0.3rem; font-variant-numeric: tabular-nums; }}
table.cm th, table.cm td {{ border: 1px solid #e2e8f0; padding: 0.3rem 0.5rem; text-align: center; min-width: 2.4rem; }}
table.cm th {{ background: #f8fafc; color: #475569; font-weight: 600; }}
table.cm td.diag {{ outline: 2px solid #16a34a; outline-offset: -2px; }}
.cm-axis {{ color: #94a3b8; font-size: 0.75rem; }}
.legend {{ display: flex; gap: 0.8rem; flex-wrap: wrap; margin: 0.3rem 0; font-size: 0.8rem; color: #475569; }}
.legend i {{ display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 2px; vertical-align: -1px; margin-right: 0.25rem; }}
</style></head>
<body>
<h1>Model tester</h1>
<div class="topnav">
  {dataset_link}<a href="/logout">Log out</a>
</div>

<div class="card">
  <label for="model">Model</label>
  <select id="model"></select>
  <small id="model-meta"></small>
</div>

<div class="card" id="perf" style="display:none;">
  <label>Held-out performance</label>
  <div id="perf-body"></div>
</div>

<div class="card">
  <label>Audio input</label>
  <div class="row" style="margin-bottom:0.8rem;">
    <input type="file" id="file" accept="audio/*">
  </div>
  <div class="row">
    <button id="rec" class="secondary" type="button">● Record</button>
    <span id="rectime"></span>
  </div>
  <audio id="player" controls hidden></audio>
</div>

<div class="row">
  <button id="run" type="button">Run inference</button>
  <span id="status"></span>
</div>

<div id="result" class="card" style="display:none;"></div>

<script>
let recorder = null, chunks = [], recordedBlob = null, recTimer = null, recStart = 0;

async function loadModels() {{
  const sel = document.getElementById('model');
  const meta = document.getElementById('model-meta');
  try {{
    const res = await fetch('/models');
    const data = await res.json();
    if (!data.ok) {{ meta.textContent = 'error: ' + data.error; return; }}
    sel.innerHTML = '';
    data.models.forEach((m, i) => {{
      const opt = document.createElement('option');
      const acc = (m.test_accuracy != null) ? ' — country acc ' + (m.test_accuracy*100).toFixed(1) + '%' : '';
      const fake = (m.fake_macro_f1 != null) ? ' — fake F1 ' + (m.fake_macro_f1*100).toFixed(1) + '%' : '';
      const tag = m.multitask ? ' [multitask]' : '';
      opt.value = m.job;
      opt.textContent = m.job + tag + acc + fake + (i === 0 ? '  (latest)' : '');
      sel.appendChild(opt);
    }});
    onModelChange();
  }} catch (e) {{ meta.textContent = 'failed to load models: ' + e; }}
}}

function updateMeta() {{
  const meta = document.getElementById('model-meta');
  meta.textContent = 'first run of a model loads its weights from GCS (~10-40s), then it stays warm.';
}}
function onModelChange() {{ updateMeta(); loadMetrics(); }}
document.getElementById('model').onchange = onModelChange;

// --- held-out performance viz (fetched per selected model) ---
const PCT = x => (x == null ? '—' : (x * 100).toFixed(1) + '%');
// value 0..1 -> blue shade for the confusion-matrix heatmap
function shade(v) {{
  const t = Math.max(0, Math.min(1, v));
  const r = Math.round(255 - t * (255 - 37));
  const g = Math.round(255 - t * (255 - 99));
  const b = Math.round(255 - t * (255 - 235));
  return 'rgb(' + r + ',' + g + ',' + b + ')';
}}

async function loadMetrics() {{
  const perf = document.getElementById('perf');
  const body = document.getElementById('perf-body');
  const job = document.getElementById('model').value;
  if (!job) {{ perf.style.display = 'none'; return; }}
  body.innerHTML = '<span class="pc-sub">loading metrics…</span>';
  perf.style.display = 'block';
  try {{
    const res = await fetch('/metrics/' + encodeURIComponent(job));
    const data = await res.json();
    if (!data.ok) {{ body.innerHTML = '<span class="error">error: ' + data.error + '</span>'; return; }}
    renderMetrics(data);
  }} catch (e) {{
    body.innerHTML = '<span class="error">failed to load metrics: ' + e + '</span>';
  }}
}}

function statCard(k, v) {{
  return '<div class="stat"><div class="stat-v">' + v + '</div><div class="stat-k">' + k + '</div></div>';
}}

function renderMetrics(data) {{
  const t = data.test || {{}}, ev = data.eval || {{}}, det = data.detail;
  let html = '';

  if (data.multitask) {{
    html += '<div class="sec-title">Real/Fake detection (held-out, speaker-disjoint)</div>';
    html += '<div class="cards">';
    html += statCard('Fake acc', PCT(t.fake_accuracy));
    html += statCard('Fake macro F1', PCT(t.fake_macro_f1));
    html += '</div>';
    const fdet = data.fake_detail;
    if (fdet && fdet.per_class) {{
      fdet.labels.forEach(l => {{
        const pc = fdet.per_class[l] || {{}};
        html += '<div class="pc-row"><div class="pc-name">' + l + '</div>' +
                '<div class="pc-track"><div class="pc-fill" style="width:' +
                  ((pc.f1 == null ? 0 : pc.f1 * 100).toFixed(1)) + '%"></div></div>' +
                '<div class="pc-val">' + PCT(pc.f1) + '</div></div>';
      }});
    }}
    html += '<div class="sec-title">Country (accent) head</div>';
  }}

  html += '<div class="cards">';
  html += statCard('Test acc', PCT(t.accuracy));
  html += statCard('Test macro F1', PCT(t.macro_f1));
  if (ev.accuracy != null) html += statCard('Val acc', PCT(ev.accuracy));
  if (t.loss != null) html += statCard('Test loss', t.loss.toFixed(3));
  html += '</div>';

  if (!det || !det.per_class) {{
    html += '<span class="pc-sub">No per-class metrics saved for this model.</span>';
    document.getElementById('perf-body').innerHTML = html;
    return;
  }}

  // --- per-class bars: F1 (and precision/recall if available) ---
  const labels = det.labels || Object.keys(det.per_class);
  const hasPR = labels.some(l => det.per_class[l] && det.per_class[l].recall != null);
  html += '<div class="sec-title">Per-country ' + (hasPR ? 'recall' : 'F1') +
          '<span class="pc-sub"> — ' + (hasPR ? 'share of that country\\'s clips predicted correctly' : 'per-class F1') + '</span></div>';
  labels.forEach(l => {{
    const pc = det.per_class[l] || {{}};
    const main = hasPR ? pc.recall : pc.f1;   // recall == per-country accuracy
    const sub = hasPR
      ? ' <span class="pc-sub">P ' + PCT(pc.precision) + ' · F1 ' + PCT(pc.f1) +
        (pc.support != null ? ' · n=' + pc.support : '') + '</span>'
      : '';
    html += '<div class="pc-row"><div class="pc-name">' + l + '</div>' +
            '<div class="pc-track"><div class="pc-fill" style="width:' +
              ((main == null ? 0 : main * 100).toFixed(1)) + '%"></div></div>' +
            '<div class="pc-val">' + PCT(main) + '</div></div>' +
            (sub ? '<div class="pc-row"><div></div><div>' + sub + '</div><div></div></div>' : '');
  }});

  // --- confusion matrix heatmap (rows=true, cols=pred), row-normalized ---
  if (det.confusion_matrix) {{
    const cm = det.confusion_matrix;
    html += '<div class="sec-title">Confusion matrix ' +
            '<span class="pc-sub">— rows = true country, cols = predicted, shaded by row %</span></div>';
    html += '<table class="cm"><thead><tr><th class="cm-axis">true \\\\ pred</th>';
    labels.forEach(l => html += '<th>' + l + '</th>');
    html += '</tr></thead><tbody>';
    cm.forEach((row, i) => {{
      const total = row.reduce((a, b) => a + b, 0) || 1;
      html += '<tr><th>' + labels[i] + '</th>';
      row.forEach((v, j) => {{
        const frac = v / total;
        const cls = (i === j) ? ' class="diag"' : '';
        const fg = frac > 0.6 ? '#fff' : '#334155';
        html += '<td' + cls + ' style="background:' + shade(frac) + ';color:' + fg +
                '" title="' + labels[i] + '→' + labels[j] + ': ' + v + ' (' + PCT(frac) + ')">' +
                v + '</td>';
      }});
      html += '</tr>';
    }});
    html += '</tbody></table>';
  }} else {{
    html += '<div class="pc-sub" style="margin-top:0.6rem;">Confusion matrix available for models trained after this update.</div>';
  }}

  document.getElementById('perf-body').innerHTML = html;
}}

// --- recording (browser MediaRecorder -> webm/opus) ---
document.getElementById('rec').onclick = async () => {{
  const btn = document.getElementById('rec');
  if (recorder && recorder.state === 'recording') {{
    recorder.stop();
    return;
  }}
  try {{
    const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = e => chunks.push(e.data);
    recorder.onstop = () => {{
      recordedBlob = new Blob(chunks, {{ type: recorder.mimeType || 'audio/webm' }});
      document.getElementById('file').value = '';
      const player = document.getElementById('player');
      player.src = URL.createObjectURL(recordedBlob);
      player.hidden = false;
      btn.textContent = '● Record';
      btn.classList.add('secondary');
      clearInterval(recTimer);
      document.getElementById('rectime').textContent = 'recorded ' + ((Date.now()-recStart)/1000).toFixed(1) + 's';
      stream.getTracks().forEach(t => t.stop());
    }};
    recorder.start();
    recStart = Date.now();
    btn.textContent = '■ Stop';
    btn.classList.remove('secondary');
    recTimer = setInterval(() => {{
      document.getElementById('rectime').textContent = ((Date.now()-recStart)/1000).toFixed(1) + 's';
    }}, 100);
  }} catch (e) {{
    document.getElementById('rectime').textContent = 'mic error: ' + e;
  }}
}};

// picking a file clears any recording
document.getElementById('file').onchange = () => {{
  recordedBlob = null;
  const f = document.getElementById('file').files[0];
  const player = document.getElementById('player');
  if (f) {{ player.src = URL.createObjectURL(f); player.hidden = false; }}
}};

document.getElementById('run').onclick = async () => {{
  const status = document.getElementById('status');
  const runBtn = document.getElementById('run');
  const file = document.getElementById('file').files[0];
  const blob = file || recordedBlob;
  if (!blob) {{ status.textContent = 'choose a file or record first'; status.className='error'; return; }}

  const fd = new FormData();
  fd.append('model', document.getElementById('model').value);
  fd.append('audio', blob, file ? file.name : 'recording.webm');

  runBtn.disabled = true;
  status.className = '';
  status.textContent = 'running... (first run loads the model, be patient)';
  try {{
    const res = await fetch('/predict', {{ method: 'POST', body: fd }});
    const data = await res.json();
    if (!data.ok) {{ status.textContent = 'error: ' + data.error; status.className='error'; return; }}
    renderResult(data);
    status.textContent = 'done (' + data.duration_s + 's audio)';
  }} catch (e) {{
    status.textContent = 'request failed: ' + e; status.className='error';
  }} finally {{
    runBtn.disabled = false;
  }}
}};

function renderResult(data) {{
  const box = document.getElementById('result');
  box.style.display = 'block';
  let html = '';
  if (data.fake) {{
    const isFake = data.fake.verdict.toLowerCase() === 'fake';
    const color = isFake ? '#c00' : '#16a34a';
    html += '<div style="font-size:1.3rem;font-weight:700;color:' + color +
            ';margin-bottom:0.6rem;">' + data.fake.verdict.toUpperCase() +
            (isFake ? ' \\u26a0\\ufe0f (synthesized voice)' : ' \\u2713') + '</div>';
    data.fake.probs.forEach(p => {{
      const pct = (p.prob*100).toFixed(1);
      html += '<div class="bar-wrap">' +
                '<div class="bar-label"><span>' + p.label + '</span><span>' + pct + '%</span></div>' +
                '<div class="bar-track"><div class="bar-fill" style="width:' + pct + '%;background:' + color + ';"></div></div>' +
              '</div>';
    }});
    html += '<label style="display:block;margin-top:0.8rem;">Accent prediction</label>';
  }} else {{
    html += '<label>Prediction</label>';
  }}
  data.predictions.forEach((p, i) => {{
    const pct = (p.prob*100).toFixed(1);
    html += '<div class="bar-wrap' + (i===0?' top':'') + '">' +
              '<div class="bar-label"><span>' + p.label + '</span><span>' + pct + '%</span></div>' +
              '<div class="bar-track"><div class="bar-fill" style="width:' + pct + '%"></div></div>' +
            '</div>';
  }});
  box.innerHTML = html;
}}

loadModels();
</script>
</body></html>
"""


@app.get("/")
@login_required
def index():
    link = (f'<a href="{DATASET_DASHBOARD_URL}">Dataset dashboard</a>'
            if DATASET_DASHBOARD_URL else "")
    return PAGE.format(dataset_link=link)


@app.get("/models")
@login_required
def models():
    try:
        return jsonify(ok=True, models=list_models())
    except Exception as exc:
        return jsonify(ok=False, error=str(exc))


@app.get("/metrics/<path:job>")
@login_required
def metrics(job: str):
    # 선택된 모델의 상세 지표(나라별 정확도/F1 + 혼동행렬)를 반환.
    try:
        return jsonify(ok=True, **get_metrics(job))
    except Exception as exc:
        return jsonify(ok=False, error=str(exc))


@app.post("/predict")
@login_required
def predict():
    job = request.form.get("model")
    if not job:
        return jsonify(ok=False, error="no model selected")
    f = request.files.get("audio")
    if f is None:
        return jsonify(ok=False, error="no audio uploaded")
    try:
        raw = f.read()
        result = run_inference(job, raw)
        return jsonify(ok=True, **result)
    except Exception as exc:
        return jsonify(ok=False, error=str(exc))


# ---------------------------------------------------------------------------
# Machine API  —  다른 서버가 X-API-Key 헤더로 호출하는 JSON 전용 엔드포인트.
# 브라우저 세션 로그인과 무관 — 위 /models, /metrics, /predict 와 로직은 같고
# 인증 방식만 다르다 (서버 간 호출은 폼 로그인을 할 수 없으므로 분리).
# ---------------------------------------------------------------------------
@app.get("/api/models")
@api_key_required
def api_models():
    try:
        return jsonify(ok=True, models=list_models())
    except Exception as exc:
        return jsonify(ok=False, error=str(exc))


@app.get("/api/metrics/<path:job>")
@api_key_required
def api_metrics(job: str):
    try:
        return jsonify(ok=True, **get_metrics(job))
    except Exception as exc:
        return jsonify(ok=False, error=str(exc))


@app.post("/api/predict")
@api_key_required
def api_predict():
    # model 생략 시 가장 최신(list_models()가 최신순으로 반환) job을 쓴다 —
    # 호출 서버가 job 이름을 몰라도 되게.
    job = request.form.get("model")
    if not job:
        models = list_models()
        if not models:
            return jsonify(ok=False, error="no trained models found under MODEL_ROOT"), 503
        job = models[0]["job"]
    f = request.files.get("audio")
    if f is None:
        return jsonify(ok=False, error="no audio uploaded (multipart field 'audio')"), 400
    try:
        raw = f.read()
        result = run_inference(job, raw)
        return jsonify(ok=True, model=job, **result)
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8766"))
    print(f"Model tester: http://{host}:{port}/  (models={MODEL_ROOT})")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
