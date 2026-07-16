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
    PORT              listen port (Cloud Run injects this; default 8766)

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
from transformers import Wav2Vec2FeatureExtractor

from config import LABELS, MODEL_NAME, SAMPLE_RATE
from model import AccentClassifier

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(32)

# 단일 사용자 로그인 정보. 데이터셋 대시보드와 동일한 기본값을 공유한다.
DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "geonah")
DASHBOARD_PASS = os.environ.get("DASHBOARD_PASS", "dmdlsldk2!")

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
                entry["test_accuracy"] = m.get("test_accuracy")
                entry["eval_accuracy"] = m.get("eval_accuracy")
                entry["macro_f1"] = m.get("test_macro_f1", m.get("eval_macro_f1"))
            except Exception:
                pass
        models.append(entry)
    return models


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
    cfg = model_dir / "label_config.json"
    if cfg.exists():
        labels = json.loads(cfg.read_text())["labels"]

    # 백본은 config로만 짓고(HF 재다운로드 없음) 우리 safetensors로 덮어쓴다.
    model = AccentClassifier(MODEL_NAME, num_labels=len(labels), pretrained=False)
    from safetensors.torch import load_file

    state = load_file(str(model_dir / "model.safetensors"))
    model.load_state_dict(state)
    model.eval()

    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    loaded = (model, fe, labels)
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
    model, fe, labels = get_model(job)
    wav = decode_audio(raw)
    inputs = fe([wav], sampling_rate=SAMPLE_RATE, return_attention_mask=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
    order = np.argsort(probs)[::-1]
    ranked = [{"label": labels[i], "prob": float(probs[i])} for i in order]
    return {"duration_s": round(wav.size / SAMPLE_RATE, 2), "predictions": ranked}


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
      const acc = (m.test_accuracy != null) ? ' — test acc ' + (m.test_accuracy*100).toFixed(1) + '%' : '';
      opt.value = m.job;
      opt.textContent = m.job + acc + (i === 0 ? '  (latest)' : '');
      sel.appendChild(opt);
    }});
    updateMeta();
  }} catch (e) {{ meta.textContent = 'failed to load models: ' + e; }}
}}

function updateMeta() {{
  const meta = document.getElementById('model-meta');
  meta.textContent = 'first run of a model loads its weights from GCS (~10-40s), then it stays warm.';
}}
document.getElementById('model').onchange = updateMeta;

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
  let html = '<label>Prediction</label>';
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


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8766"))
    print(f"Model tester: http://{host}:{port}/  (models={MODEL_ROOT})")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
