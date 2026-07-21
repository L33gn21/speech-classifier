"""Central configuration for the accent/country classifier.

Data source is the curated GCS pool documented in DATASET.md (us-west2 rebuild):

    gs://qi-ucsd-speech-usw2/curated/<CC>/manifest.csv   (fname,source,speaker,gender,age,accent)
    gs://qi-ucsd-speech-usw2/curated/<CC>/audio/<fname>

The country label is the folder name (``<CC>``), not a column.

Paths are environment-driven so the same code runs locally and on Vertex AI:

- Locally, sensible defaults under the repo root are used.
- On Vertex AI Custom Training, buckets are FUSE-mounted at ``/gcs/<bucket>``
  and the job output dir is provided via ``AIP_MODEL_DIR`` (a ``gs://`` URI).
  Set ``CV_CURATED_ROOT`` to the ``gs://`` (or mounted ``/gcs``) curated path.

Env vars (all optional):
    CV_CURATED_ROOT   root holding <CC>/manifest.csv and <CC>/audio/
                      (default: <repo>/curated)
    CV_OUTPUT_DIR     where the trained model is written
                      (default: AIP_MODEL_DIR if set, else <repo>/outputs/classifier)
    CV_MODEL_NAME     pretrained wav2vec2 backbone (default facebook/wav2vec2-base)
"""
# 억양/국가 분류기 전역 설정 파일.
# 데이터는 DATASET.md 에 기술된 curated GCS 풀을 사용한다 (us-west2 재구축):
#   gs://qi-ucsd-speech-usw2/curated/<CC>/manifest.csv  (fname,source,speaker,gender,age,accent)
#   gs://qi-ucsd-speech-usw2/curated/<CC>/audio/<fname>
# 국가 라벨은 컬럼이 아니라 폴더 이름(<CC>)이다.
# 경로들을 환경변수로 제어하여 로컬 환경과 Vertex AI 환경에서
# 동일한 코드가 그대로 동작하도록 만든다.
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution helpers
# 경로 해석 관련 헬퍼 함수들
# ---------------------------------------------------------------------------
# config.py lives at classifier/src/config.py -> repo root is three levels up.
# 이 파일은 classifier/src/config.py 에 위치하므로, 저장소(repo) 루트는
# 상위로 3단계 올라간 곳이다.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def gcs_to_fuse(path: str) -> str:
    """Translate a ``gs://bucket/x`` URI to its Vertex AI FUSE mount ``/gcs/bucket/x``.

    Vertex AI Custom Training auto-mounts accessible buckets under ``/gcs``.
    Non-``gs://`` paths are returned unchanged.
    """
    # "gs://버킷/경로" 형태의 GCS URI를,
    # Vertex AI에서 자동으로 마운트되는 실제 파일시스템 경로
    # "/gcs/버킷/경로" 로 변환한다.
    # gs:// 로 시작하지 않는 일반 경로는 그대로 반환한다(로컬 실행 대비).
    if path.startswith("gs://"):
        return "/gcs/" + path[len("gs://"):]
    return path


def _env_path(name: str, default: Path) -> Path:
    # 환경변수 `name`이 설정되어 있으면 그 값을(GCS 경로라면 FUSE 경로로 변환하여) 사용하고,
    # 없으면 넘겨받은 기본 경로(default)를 그대로 사용한다.
    raw = os.environ.get(name)
    if raw:
        return Path(gcs_to_fuse(raw))
    return default


# ---------------------------------------------------------------------------
# Paths (env-overridable)
# 경로 설정 (환경변수로 재정의 가능)
# ---------------------------------------------------------------------------
# curated 풀의 루트 디렉터리. 하위에 <CC>/manifest.csv 와 <CC>/audio/ 가 있다.
CURATED_ROOT = _env_path("CV_CURATED_ROOT", REPO_ROOT / "curated")

# train.csv / val.csv / test.csv 분할 매니페스트가 저장되는 디렉터리.
# curated/ 원본은 절대 건드리지 않고, 분할 결과만 여기(기본 outputs 하위)에 기록한다.
MANIFEST_DIR = _env_path(
    "CV_MANIFEST_DIR", REPO_ROOT / "outputs" / "classifier" / "manifests"
)

# 출력 디렉터리 기본값 결정 순서:
# 1) Vertex AI가 자동으로 넘겨주는 AIP_MODEL_DIR 환경변수
# 2) 그것도 없으면 로컬 저장소의 outputs/classifier
_default_output = os.environ.get("AIP_MODEL_DIR") or str(REPO_ROOT / "outputs" / "classifier")
# 학습된 모델(가중치, 설정, 로그 등)이 저장될 최종 출력 디렉터리.
OUTPUT_DIR = _env_path("CV_OUTPUT_DIR", Path(gcs_to_fuse(_default_output)))

# ---------------------------------------------------------------------------
# Label space
# 레이블(분류 클래스) 정의
# ---------------------------------------------------------------------------
# curated 풀에 실제로 채워져 있는 국가 클래스(폴더 이름).
# us-west2 재구축 스코프: GLOBE + SAA 만으로 견고하게 채워지는 6개 클래스.
#   US/UK/CA/AU/IN = GLOBE(볼륨) + SAA(화자 다양성), 하이브리드 라벨은 국가.
#   CN = GLOBE 홍콩 + SAA 만다린/광둥(모국어 축) — 6개 중 가장 작은 클래스(하한).
# (NG/JP/CN 원전용 소스 AfriSpeech/SpeechOcean762는 이 버킷에 없음 — DATASET.md 참고)
# Fixed order -> integer label id. Keep stable; the trained head depends on it.
# 순서는 반드시 고정되어야 한다. 학습된 분류 헤드의 출력 인덱스가 이 순서에
# 의존하기 때문에, 순서를 바꾸면 기존 모델과 어긋나게 된다.
LABELS: list[str] = ["US", "UK", "CA", "AU", "IN", "CN"]
LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: name for name, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

# ---------------------------------------------------------------------------
# Spoof / fake-detection axis (multi-task real/fake head)
# fake(합성 음성) 탐지 축 — 국가와 별개의 두 번째 라벨 축(멀티태스크 헤드)
# ---------------------------------------------------------------------------
# spoof 코퍼스(ASVspoof 2019 LA)는 국가 curated 풀과 분리된 별도 프리픽스에 있다
# (레이아웃·스키마가 다르므로 국가 로더는 이걸 스캔하지 않는다 — DATASET.md §10).
# 기본값을 CURATED_ROOT 의 "형제" 경로로 두면, Vertex 에서 CV_CURATED_ROOT 가
# gs://bucket/curated 로 주입될 때 spoof 도 /gcs/bucket/curated_spoof/... 로 자동
# 해석된다(추가 env 주입 불필요). CV_SPOOF_ROOT 로 따로 재정의도 가능.
SPOOF_ROOT = _env_path(
    "CV_SPOOF_ROOT", CURATED_ROOT.parent / "curated_spoof" / "asvspoof2019_la"
)
# ASVspoof 프로토콜 스플릿 — 반드시 보존(eval 은 train/dev 에 없는 미지 공격 A07–A19).
SPOOF_SPLITS: list[str] = ["train", "dev", "eval"]

# real/fake 헤드 레이블(이진). 순서 고정: real=0, fake=1 (학습된 fake 헤드가 의존).
FAKE_LABELS: list[str] = ["real", "fake"]
FAKE2ID: dict[str, int] = {name: i for i, name in enumerate(FAKE_LABELS)}
ID2FAKE: dict[int, str] = {i: name for name, i in FAKE2ID.items()}
NUM_FAKE_LABELS = len(FAKE_LABELS)

# country 헤드에서 국가 라벨이 없는 클립(=spoof 코퍼스)을 손실 계산에서 제외하기
# 위한 sentinel. torch.nn.functional.cross_entropy(ignore_index=...) 규약과 동일.
COUNTRY_IGNORE_INDEX = -100

# ---------------------------------------------------------------------------
# Audio
# 오디오 관련 설정
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000  # wav2vec2가 요구하는 샘플링 레이트(16kHz)
MAX_DURATION_S = 8.0  # crop cap; collator pads to per-batch max up to this
# 최대 오디오 길이(초). 이보다 긴 클립은 이 길이로 잘라낸다(crop).
# 배치를 만들 때는 collator가 배치 내 최대 길이까지만 패딩하므로,
# 이 값은 어디까지나 "상한선" 역할이다.
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION_S)  # 초 단위 길이를 샘플 개수로 환산

# ---------------------------------------------------------------------------
# Data preparation
# 데이터 전처리(준비) 관련 설정
# ---------------------------------------------------------------------------
# 큰 클래스는 클래스당 상한(TARGET_PER_CLASS)까지만 언더샘플링한다. curated 풀이
# 5k-scale로 재구축(2026-07-16, spec_5k.json)되어 US/UK/CA는 ~6k, AU/IN은 4~4.8k,
# CN(최소 클래스, 홍콩영어 상한)은 ~1.17k이다. 상한을 5000으로 두면 큰 3개 클래스는
# 5000으로 캡되고 나머지는 전량 사용된다. 불균형은 macro-F1 + class weighting으로 흡수.
TARGET_PER_CLASS = 5000           # balanced under-sampling cap per class
# 화자당 클립 상한. 일부 소스(예: SpeechOcean762 CN)는 한 화자(id)에 수백 개
# 클립이 묶여 있어, 그대로 두면 한 화자가 클래스와 분할을 통째로 지배해버린다.
# 화자당 클립을 이 값으로 먼저 제한해 클래스/분할이 다양한 화자로 채워지게 한다.
# (DATASET.md 의 원본 큐레이션도 소스별 ≤30 clips/speaker 관례를 쓴다.)
MAX_CLIPS_PER_SPEAKER = 20        # per-speaker clip cap (applied before class cap)
VAL_FRACTION = 0.15               # speaker-level validation holdout fraction
TEST_FRACTION = 0.15              # speaker-level test holdout fraction
SEED = 42                         # 재현성을 위한 랜덤 시드 고정값

# ---------------------------------------------------------------------------
# Model
# 모델 관련 설정
# ---------------------------------------------------------------------------
# 사용할 사전학습 백본 모델 이름. 환경변수 CV_MODEL_NAME으로 다른 모델을
# 지정할 수 있으며, 기본값은 facebook/wav2vec2-base.
MODEL_NAME = os.environ.get("CV_MODEL_NAME", "facebook/wav2vec2-base")
