안녕! 지금은 2026년 7월 9일 목요일 오후 2시 52분(PDT)이야.

네가 기존에 짰던 탄탄한 파이썬 백엔드(PyTorch 모델) 기획에, 이번에 깃허브에 올린 **'Google AI Studio Edition' (React + Node.js 풀스택 웹 애플리케이션)** 내용을 완벽하게 하나로 융합해서 README.md를 다시 작성해 줄게.

어디 내놔도 손색없는 프로페셔널한 오픈소스 프로젝트 스타일로 구조화했으니까, 이 내용 그대로 복붙해서 쓰면 돼!

---

# Speech Classifier System (Google AI Studio Edition)

음성 기반 범죄 수사 보조 도구이자, 실시간으로 AI 합성 음성을 탐지하고 사람의 영어 억양을 분류하도록 설계된 풀스택 웹 애플리케이션입니다. 발화를 2단계 핵심 파이프라인으로 처리하며, Web Audio API를 활용한 실시간 분석 기능을 제공합니다.

`발화 → [Detector] AI 판별 시 종료 / 사람 판별 시 → [Classifier] 언어 감지 및 억양(국적 단서) 추정`

## ✨ 주요 기능 (Features)

* **Stage 1: AI vs Human Detection (Detector)**

* 오디오를 분석하여 해당 음성이 AI에 의해 생성된 것인지 실제 사람의 음성인지 판별합니다.


* 한국어(Korean), 영어(English) 등 언어별 프로파일링(Language-specific profiling)을 적용하여 교차 언어 오분류(cross-lingual misclassification)를 방지합니다.




* **Stage 2: Accent Probability Estimation (Classifier)**

* 음성이 사람의 것으로 판별되고 언어가 영어일 경우, 미국(US), 영국(England), 인도(Indian), 호주(Australia) 억양에 대한 확률을 추정합니다.


* 영어 지역 방언(English regional dialects)에 특별히 최적화되어 있습니다.




* **Target Speaker Isolation (주요 화자 격리)**

* Web Audio API 제약 조건(constraints)을 사용하여 배경 소음을 필터링하고 주요 화자에게 특별히 초점을 맞춥니다.




* **Real-Time Visualization (실시간 시각화)**

* 라이브 오디오 파형(waveform)을 시각화하여 보여줍니다.


* 시간의 흐름에 따른 억양 확률 변화를 동적 선형 차트(Dynamic line charts)로 추적하여 Accent Drift(억양 표류) 현상을 시각화합니다.





## 📂 프로젝트 구조 (Project Structure)

프로젝트는 프론트엔드/웹 서버와 파이썬 모델 훈련 파이프라인으로 구성되어 있습니다.

```text
├── src/
│   ├── App.tsx             # 메인 React 애플리케이션 (UI, 오디오 시각화 및 분석 로직)[cite: 3]
│   ├── app.py              # Python 모델 통합 진입점 (CLI 단일 파일 분석용)
│   ├── detector/           # 1단계: AI 합성 음성 탐지 모델 및 데이터셋 준비[cite: 3]
│   │   ├── model.py        # resnet18(1ch) on log-mel → real/fake 판별
│   │   └── ...
│   └── classifier/         # 2단계: 억양 확률 분류 모델 (wav2vec2 기반)[cite: 3]
│       ├── model.py        # wav2vec2 + frame-level head 적용
│       └── ...
├── server.ts               # Vite 미들웨어가 구성된 Express 백엔드 (개발 및 프로덕션 서빙용)[cite: 3]
├── data/                   # 훈련용 오디오 데이터셋 디렉터리 (detector & classifier)
└── outputs/                # 학습된 모델 가중치 (.pt) 저장소

```


```

### 2. Python Backend & Model Training 실행 (개발자용)

단일 `.venv` (Python 3.13 + torch 2.9.1+cu126) 환경에서 두 모델의 학습, 추론, 데이터 준비를 모두 수행합니다.

```bash
# 가상환경 구성 (uv 활용)
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126

# [1단계] Detector 학습 및 추론
.venv/bin/python src/detector/train.py                  # 모델 학습
.venv/bin/python src/detector/inference.py clip.wav     # 단일 추론

# [2단계] Classifier 학습 및 추론
.venv/bin/python src/classifier/prepare_data.py         # 매니페스트 생성
.venv/bin/python src/classifier/train.py --epochs 8 ... # 모델 학습
.venv/bin/python src/classifier/infer.py clip.mp3       # 단일 추론

# [통합 CLI 모드 실행]
.venv/bin/python src/app.py path/to/clip.wav

```

> **주의:** 현재 `data/detector`에는 fake 데이터만 존재하므로, 학습 전 `real/` 폴더를 반드시 채워야 합니다.

## 📝 모델 설계 메모 (Design Notes)

탐지기(Detector) 데이터 균형:** Real/Fake의 화자 및 문장 분포가 겹치게 맞추어 데이터셋 간의 차이가 아닌 "합성 아티팩트(Artifacts)" 자체를 학습하도록 유도합니다. (여러 보코더 혼합 사용 권장)
분류기(Classifier) 데이터 분리:** 화자의 목소리 특성을 암기하는 것을 방지하기 위해 `client_id` 기준으로 Train/Test 셋을 분리합니다.
Frame-level 보존 (Level 2 대비):** Head를 프레임마다 적용하여 `frame_logits [B,T,C]`를 생성하고 시간축 masked-mean으로 발화(utterance) 로짓을 도출합니다. 이를 통해 수학적 손실 없이 구간별 억양 히트맵 출력을 무료로 확보할 수 있습니다.
라벨링: 억양 라벨은 `accents` 서술형 문자열을 정확히 매칭(혼합 `a|b` 행 제외)하여 매핑합니다.
