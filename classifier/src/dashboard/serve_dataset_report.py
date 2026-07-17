"""Live dataset dashboard — a small web server with a Refresh button.

Wraps inspect_dataset.py's report in a page that re-reads the curated
manifests from GCS on demand (button click), instead of re-running the CLI
script by hand every time the dataset changes.

Config is read from env vars so the same module works both as a local
script (`python serve_dataset_report.py`) and as a gunicorn app on Cloud Run
(`gunicorn serve_dataset_report:app`), where nothing calls main():

    DATASET_ROOT     curated/ root, gs:// URI or local dir (default: bucket)
    DATASET_CLASSES  comma-separated class list (default: US,UK,IN,NG,KR)
    PORT             listen port (Cloud Run injects this; default 8765)

Usage:
    python serve_dataset_report.py
    DATASET_ROOT=gs://qi-ucsd-speech-us/curated python serve_dataset_report.py

Then open http://127.0.0.1:8765/ and click "Refresh" after the curated
dataset changes.
"""
# 데이터셋 대시보드를 "새로고침 버튼"이 있는 웹 서버로 띄우는 스크립트.
#
# inspect_dataset.py는 실행할 때마다 정적 HTML 파일을 새로 만드는 1회성
# 스크립트였다. 이 스크립트는 그 로직(collect_manifests / render_body)을
# 그대로 재사용하되, Flask로 감싸서 브라우저에서 버튼을 누를 때마다
# GCS 매니페스트를 다시 읽고 화면을 그 자리에서 갱신한다.
#
# 설정은 환경변수로 읽는다 — 로컬 스크립트(`python serve_dataset_report.py`)로도,
# Cloud Run 위의 gunicorn 앱(`gunicorn serve_dataset_report:app`, main()을
# 호출하지 않음)으로도 동일한 모듈이 그대로 동작하게 하기 위함:
#
#     DATASET_ROOT     curated/ 루트, gs:// URI 또는 로컬 경로 (기본: 버킷)
#     DATASET_CLASSES  콤마로 구분한 클래스 목록 (기본: US,UK,IN,NG,KR)
#     PORT             리스닝 포트 (Cloud Run이 주입; 기본 8765)
#
# 사용 예시:
#     python serve_dataset_report.py
#     DATASET_ROOT=gs://qi-ucsd-speech-us/curated python serve_dataset_report.py
#
# 브라우저에서 http://127.0.0.1:8765/ 접속 후, 데이터셋이 바뀔 때마다
# "Refresh" 버튼을 누르면 다시 읽어와 그려준다.
from __future__ import annotations

import datetime
import os
import threading
from functools import wraps

from flask import Flask, jsonify, redirect, request, session, url_for

from inspect_dataset import DEFAULT_CLASSES, DEFAULT_ROOT, SHARED_STYLE, collect_manifests, render_body

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(32)

# 단일 사용자 로그인 정보. 배포 시 DASHBOARD_USER / DASHBOARD_PASS 환경변수로 덮어쓸 수 있음.
DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "geonah")
DASHBOARD_PASS = os.environ.get("DASHBOARD_PASS", "dmdlsldk2!")


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


LOGIN_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Login — Dataset dashboard</title>
<style>""" + SHARED_STYLE + """
body {{ display: flex; align-items: center; justify-content: center; padding: 1.5rem; }}
.login-card {{ padding: 2.4rem 2.6rem; min-width: 300px; display: flex; flex-direction: column; gap: 1rem; }}
.login-badge {{ width: 40px; height: 40px; border-radius: 11px; display: flex; align-items: center;
                justify-content: center; font-size: 1.2rem;
                background: linear-gradient(135deg, var(--accent), var(--accent-2)); }}
.login-card h1 {{ font-size: 1.15rem; margin: .2rem 0 0; }}
.login-card p {{ margin: 0; color: var(--muted); font-size: .85rem; }}
input {{ padding: .65rem .8rem; font-size: .95rem; border: 1px solid var(--border); border-radius: 8px;
         background: var(--bg); color: var(--ink); outline: none; transition: border-color .15s; }}
input:focus {{ border-color: var(--accent); }}
button {{ padding: .65rem; font-size: .95rem; font-weight: 600; cursor: pointer; border: none; border-radius: 8px;
          background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #fff;
          box-shadow: 0 6px 16px -6px color-mix(in srgb, var(--accent) 60%, transparent); }}
button:hover {{ filter: brightness(1.06); }}
.error {{ color: #ef4444; margin: 0; font-size: .85rem; }}
</style></head>
<body>
<form class="card login-card" method="post" action="/login">
<span class="login-badge">🎙️</span>
<h1>Dataset dashboard</h1>
<p>Sign in to view the curated speech-accent corpus.</p>
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

# 서버가 마지막으로 그린 리포트 상태. 여러 요청이 동시에 들어와도 안전하도록 _lock으로 보호.
_state = {
    "root": os.environ.get("DATASET_ROOT", DEFAULT_ROOT),
    "classes": os.environ.get("DATASET_CLASSES", ",".join(DEFAULT_CLASSES)).split(","),
    "body": "<p>Loading...</p>",
    "error": None,
}
_lock = threading.Lock()


def _regenerate() -> None:
    # GCS에서 매니페스트를 다시 읽어 리포트 본문(body)을 새로 만든다.
    # 실패해도 서버를 죽이지 않고 에러 메시지를 상태에 담아 화면에 보여준다.
    try:
        dfs = collect_manifests(_state["root"], _state["classes"])
        if not dfs:
            with _lock:
                _state["error"] = "No manifests found under root."
            return
        body = render_body(_state["root"], dfs)
        with _lock:
            _state["body"] = body
            _state["error"] = None
    except Exception as exc:
        with _lock:
            _state["error"] = str(exc)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Curated dataset dashboard</title>
<style>""" + SHARED_STYLE + """
.topbar {{ display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap;
           gap: .8rem; margin-bottom: .3rem; }}
.title-group {{ display: flex; align-items: center; gap: .7rem; }}
.badge {{ width: 38px; height: 38px; border-radius: 10px; flex: none; display: flex; align-items: center;
          justify-content: center; font-size: 1.05rem;
          background: linear-gradient(135deg, var(--accent), var(--accent-2)); }}
.actions {{ display: flex; align-items: center; gap: .7rem; }}
#refresh {{ font-size: .88rem; font-weight: 600; padding: .5rem 1.1rem; cursor: pointer; border: none;
            border-radius: 8px; color: #fff; background: linear-gradient(135deg, var(--accent), var(--accent-2));
            box-shadow: 0 6px 16px -6px color-mix(in srgb, var(--accent) 60%, transparent);
            transition: filter .15s, transform .15s; }}
#refresh:hover {{ filter: brightness(1.06); }}
#refresh:disabled {{ opacity: .6; cursor: default; }}
#refresh.spin::before {{ content: "↻ "; display: inline-block; animation: spin .8s linear infinite; }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.logout-link {{ color: var(--muted); font-size: .85rem; text-decoration: none; border: 1px solid var(--border);
                 border-radius: 8px; padding: .5rem .9rem; background: var(--surface); }}
.logout-link:hover {{ color: var(--ink); border-color: var(--accent); }}
#status {{ color: var(--muted); font-size: .82rem; display: block; margin: .3rem 0 1rem; }}
.error {{ color: #ef4444; }}
</style></head>
<body>
<div class="topbar">
  <div class="title-group">
    <span class="badge">🎙️</span>
    <div>
      <h1>Curated dataset dashboard</h1>
      <p class="subtitle">Speech-accent corpus &middot; live from GCS</p>
    </div>
  </div>
  <div class="actions">
    <button id="refresh">Refresh</button>
    <a class="logout-link" href="/logout">Log out</a>
  </div>
</div>
<span id="status"></span>
<div id="report">{body}</div>
<script>
async function doRefresh() {{
  const status = document.getElementById('status');
  const btn = document.getElementById('refresh');
  btn.disabled = true;
  btn.classList.add('spin');
  status.textContent = 'Refreshing…';
  status.className = '';
  try {{
    const res = await fetch('/refresh', {{ method: 'POST' }});
    const data = await res.json();
    if (data.ok) {{
      document.getElementById('report').innerHTML = data.body;
      status.textContent = 'Updated ' + data.generated;
    }} else {{
      status.textContent = 'Error: ' + data.error;
      status.className = 'error';
    }}
  }} catch (e) {{
    status.textContent = 'Request failed: ' + e;
    status.className = 'error';
  }} finally {{
    btn.disabled = false;
    btn.classList.remove('spin');
  }}
}}
document.getElementById('refresh').onclick = doRefresh;
</script>
</body></html>
"""


@app.get("/")
@login_required
def index():
    with _lock:
        body = _state["body"]
        error = f'<p class="error">{_state["error"]}</p>' if _state["error"] else ""
    return PAGE.format(body=error + body)


@app.post("/refresh")
@login_required
def refresh():
    # 버튼 클릭 시 프론트에서 호출하는 엔드포인트. 매니페스트를 다시 읽고
    # 새 body HTML을 JSON으로 돌려주면, 프론트가 innerHTML만 교체한다(페이지 새로고침 없음).
    _regenerate()
    with _lock:
        if _state["error"]:
            return jsonify(ok=False, error=_state["error"])
        return jsonify(
            ok=True,
            body=_state["body"],
            generated=datetime.datetime.now().isoformat(timespec="seconds"),
        )


# 서버 기동 시 최초 1회 생성해둔다 (첫 화면부터 데이터가 보이도록). gunicorn은
# main()을 부르지 않고 이 모듈을 import만 하므로, 모듈 레벨에서 실행해야 한다.
_regenerate()


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8765"))
    print(f"Dataset dashboard: http://{host}:{port}/  (root={_state['root']})")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
