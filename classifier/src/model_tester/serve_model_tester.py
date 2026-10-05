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

# Single-user login. Shares the same defaults as the dataset dashboard.
DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "geonah")
DASHBOARD_PASS = os.environ.get("DASHBOARD_PASS", "dmdlsldk2!")

# Static API key that other servers use to call /api/*. Demo-grade — only the header
# value is compared, with no rotation/scoping. deploy.sh injects it as an env var.
API_KEY = os.environ.get("API_KEY", "dev-key-change-me")

# GCS prefix where the training jobs accumulate. Each job keeps model.safetensors /
# label_config.json / preprocessor_config.json / final_metrics.json under
# <MODEL_ROOT>/<JOB_NAME>/model/.
MODEL_ROOT = os.environ.get(
    "MODEL_ROOT", "gs://qi-ucsd-speech-us/outputs/classifier"
).rstrip("/")

# Link back to the dataset dashboard (shown at the top when set).
DATASET_DASHBOARD_URL = os.environ.get("DATASET_DASHBOARD_URL", "")

# Maximum upload size (25 MB). Enough for short test clips; blocks excessive uploads.
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# ---------------------------------------------------------------------------
# GCS helpers
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
    # A single storage.Client is reused instead of creating a new one per thread.
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
    bucket_name, prefix = _split_gs(MODEL_ROOT)
    prefix = prefix.rstrip("/") + "/"
    bucket = _client().bucket(bucket_name)

    # List with a delimiter to get only the "folders" (job names) directly under the prefix.
    it = _client().list_blobs(bucket_name, prefix=prefix, delimiter="/")
    list(it)  # prefixes are only populated once the iteration has finished
    job_prefixes = sorted(it.prefixes, reverse=True)

    models: list[dict] = []
    for jp in job_prefixes:
        job = jp[len(prefix):].strip("/")
        weights = bucket.blob(f"{prefix}{job}/model/model.safetensors")
        if not weights.exists():
            continue  # expose only jobs that finished training and saving
        entry: dict = {"job": job}
        metrics_blob = bucket.blob(f"{prefix}{job}/model/final_metrics.json")
        if metrics_blob.exists():
            try:
                m = json.loads(metrics_blob.download_as_bytes())
                # Multitask jobs use the "test_country_accuracy"/"test_fake_macro_f1" keys instead of
                # "test_accuracy" — both are accepted as fallbacks.
                multitask = bool(m.get("train_config", {}).get("multitask")) or "test_fake_macro_f1" in m
                entry["multitask"] = multitask
                entry["test_accuracy"] = m.get("test_accuracy", m.get("test_country_accuracy"))
                entry["eval_accuracy"] = m.get("eval_accuracy", m.get("eval_country_accuracy"))
                entry["macro_f1"] = m.get("test_macro_f1", m.get("test_country_macro_f1"))
                entry["fake_macro_f1"] = m.get("test_fake_macro_f1")
                # Flag whether the model has a per-country breakdown (the dropdown stays light; the
                # detailed metrics themselves are fetched separately from /metrics/<job> on selection).
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
    bucket_name, prefix = _split_gs(MODEL_ROOT)
    prefix = prefix.rstrip("/") + "/"
    bucket = _client().bucket(bucket_name)
    blob = bucket.blob(f"{prefix}{job}/model/final_metrics.json")
    if not blob.exists():
        return {"job": job, "metrics": None, "detail": None}
    m = json.loads(blob.download_as_bytes())
    multitask = bool(m.get("train_config", {}).get("multitask")) or "test_fake_macro_f1" in m

    def summary(split: str) -> dict:
        # Multitask jobs use "{split}_country_*" keys for the country metrics (see list_models).
        return {
            "accuracy": m.get(f"{split}_accuracy", m.get(f"{split}_country_accuracy")),
            "macro_f1": m.get(f"{split}_macro_f1", m.get(f"{split}_country_macro_f1")),
            "loss": m.get(f"{split}_loss"),
            "fake_accuracy": m.get(f"{split}_fake_accuracy"),
            "fake_macro_f1": m.get(f"{split}_fake_macro_f1"),
        }

    # Detail block: newer models use it as-is; older ones get it synthesized from f1 scalars.
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

    # Per-class (real/fake) F1 of the fake head — only multitask jobs have it.
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
# Model cache
# ---------------------------------------------------------------------------
# job name -> (model, feature_extractor, labels). Downloaded from GCS and loaded into
# memory on the first request only (slow); reused while the instance stays alive (fast).
_models: dict[str, tuple] = {}
_model_lock = threading.Lock()


def _download_model_dir(job: str, dst: Path) -> None:
    # Download only the files needed for inference from the selected job's model/ directory
    # to the container's temp disk (training intermediates such as checkpoint-*/ are excluded
    # — unnecessary for inference and large).
    bucket_name, prefix = _split_gs(MODEL_ROOT)
    prefix = prefix.rstrip("/") + "/"
    bucket = _client().bucket(bucket_name)
    wanted = [
        "model.safetensors",
        "label_config.json",
        "preprocessor_config.json",
        # Without model_config.json, load_from_dir always builds the legacy default structure
        # (country only, fake_head=False) — multitask/attentive models require this file.
        "model_config.json",
    ]
    dst.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        blob = bucket.blob(f"{prefix}{job}/model/{name}")
        if blob.exists():
            blob.download_to_filename(str(dst / name))


def get_model(job: str):
    # Return the job's model from the cache; otherwise download it from GCS, load and cache it.
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

    # The backbone is built from the config only (no HF re-download) and overwritten with our
    # safetensors. load_from_dir reads model_config.json and builds the same
    # backbone/head/fake_head structure as in training (older checkpoints fall back to the
    # legacy default = country only).
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
# Audio decoding
# ---------------------------------------------------------------------------
def decode_audio(raw: bytes) -> np.ndarray:
    """Decode arbitrary audio bytes (wav/mp3/webm/ogg/m4a...) to float32 mono
    16 kHz using ffmpeg. Browser MediaRecorder emits webm/opus, so we lean on
    ffmpeg rather than torchaudio/soundfile to cover every container.
    """
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
# Auth  —  login (same pattern as the dataset dashboard)
# ---------------------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def api_key_required(view):
    # For machine-facing routes that authenticate via the X-API-Key header, not the session login.
    @wraps(view)
    def wrapped(*args, **kwargs):
        key = request.headers.get("X-API-Key", "")
        if key != API_KEY:
            return jsonify(ok=False, error="invalid or missing X-API-Key header"), 401
        return view(*args, **kwargs)

    return wrapped


# CORS for external services (the hackathon frontend) that call /api/* from the browser.
# Session-login routes are same-origin only and out of scope — only /api/* gets the headers.
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
    # Preflight requests arrive without an API key — return just 204 without auth; the
    # after_request hook above adds the headers.
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
# Page  —  main page
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
    # Return the selected model's detailed metrics (per-country accuracy/F1 + confusion matrix).
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
# Machine API  —  JSON-only endpoints that other servers call with the X-API-Key header.
# Independent of the browser session login — same logic as /models, /metrics, /predict
# above; only the authentication differs (server-to-server calls cannot use a form login).
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
    # When model is omitted, the newest job is used (list_models() returns newest first) —
    # so the calling server does not need to know the job name.
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
