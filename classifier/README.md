# 억양(국적) 분류기 — accent classifier

사람 영어 음성의 **억양권(국적 단서)** 을 추정하는 분류기.
사전학습 오디오 인코더(`wav2vec2-base`) + 분류 헤드 파인튜닝. 클래스별 softmax
확률을 그대로 "억양 근접도 퍼센트"로 사용한다 (예: US 30%, UK 45%, IN 10%, ...).

- 학습 클래스(현재): **`US`, `UK`, `CA`, `AU`, `IN`, `CN`** — 6클래스(us-west2
  재구축, GLOBE+SAA). US/UK/CA/AU/IN은 국가 라벨이고, CN은 성격이 다른 L2 억양
  (GLOBE 홍콩 English + SAA 만다린/광둥 L1)이자 가장 작은 클래스다
  (`DATASET.md` §3/§5, `src/config.py` `LABELS`). NG/JP는 이 버킷에 미구축.
- 데이터셋의 실제 저장 위치·레이아웃·큐레이션·주의사항은 **[`DATASET.md`](./DATASET.md)** 가
  소스 오브 트루스다. 학습 전에 반드시 그쪽 캐비엇(채널 혼입, 화자분리)을 읽는다.

> **운영 규칙은 [`CLAUDE.md`](./CLAUDE.md) 를 본다.** 이 프로젝트는 **gcloud-first** 다:
> 데이터·모델은 GCS/Vertex에서 관리하고, 로컬은 소스코드·환경변수·문서만 둔다.
> 데이터 다운로드는 로컬 경유 없이 Compute Engine VM→버킷 직접, 학습은 T4/us-west2.

---

## 파이프라인 (3단계 코드, 로컬/Vertex 공용)

```
1. 데이터셋 수집   prepare_data.py   curated/<CC>/manifest.csv → 균형·화자분리 split
2. 모델 설정       model.py          wav2vec2 backbone + frame-level 헤드
3. 학습 / 테스트   train.py / evaluate.py
```

학습 데이터는 이미 버킷의 `gs://qi-ucsd-speech-usw2/curated/` 에 있다(VM→버킷으로
빌드됨). `prepare_data.build_splits()` 가 이 국가별 `manifest.csv` 들로부터
클래스 균형·**화자분리** train/val/test 분할을 만든다. `train.py` 가 잡 시작 시
이걸 직접 호출하므로 **매니페스트를 따로 업로드하는 단계는 없다.**

---

## 디렉터리

```
classifier/
  src/
    config.py        # 중앙 설정 (경로는 env로 오버라이드 → 로컬/GCS 겸용)
    prepare_data.py  # 1단계: curated/<CC>/manifest.csv → train/val/test 분할
    dataset.py       # 오디오 로딩 + collator (wav2vec2 feature extractor)
    model.py         # 2단계: 모델 정의 (frame-level 출력 보존 → Level 2 대비)
    train.py         # 3단계: HuggingFace Trainer 학습 (build_splits 직접 호출)
    evaluate.py      # 3단계: 홀드아웃 테스트 (accuracy/F1/혼동행렬)
    infer.py         # 단일 오디오 → 억양 퍼센트 (+ --plot 프레임 히트맵)
    dashboard/       # 데이터셋 감사 리포트 서빙 (Cloud Run 데모)
  Dockerfile         # Vertex AI 학습 컨테이너 (CUDA torch + ffmpeg)
  gcloud/
    env.example.sh   # 프로젝트 값 템플릿 (→ env.sh 로 복사, gitignore)
    check_data.sh    # 사전점검: 버킷 curated 풀 존재/클립수 확인
    build_and_push.sh# 2단계: Cloud Build로 이미지 빌드/푸시
    submit_job.sh    # 3단계: Vertex AI Custom Job 제출 (T4)
    rebalance_in.py  # IN 클래스 재균형 유틸 (DATASET.md §3 참고)
  reports/           # 작업 보고서(작업 로그) — 지속적 발전 기록. reports/README.md 참고
```

경로는 `config.py`가 환경변수로 결정한다. 미설정 시 리포 루트 기준 로컬 기본값을
쓰고, Vertex에서는 FUSE 마운트(`/gcs/<bucket>`)와 `AIP_MODEL_DIR`을 사용한다.
`gs://bucket/...` 값을 넣으면 자동으로 `/gcs/bucket/...` FUSE 경로로 변환된다.

| env | 의미 | 기본값 |
|-----|------|--------|
| `CV_CURATED_ROOT` | `curated/<CC>/manifest.csv`+`audio/` 위치 | `<repo>/curated` |
| `CV_MANIFEST_DIR` | 생성된 `train/val/test.csv` 저장 위치 | `<repo>/outputs/classifier/manifests` |
| `CV_OUTPUT_DIR` | 모델 산출물 | `AIP_MODEL_DIR` 있으면 그것, 없으면 `<repo>/outputs/classifier` |
| `CV_MODEL_NAME` | backbone | `facebook/wav2vec2-base` |

---

## Vertex AI 실행 (gcloud) — 기본 경로

이 프로젝트의 **표준 학습 경로**다. 하드웨어는 **T4 ×1 / `n1-standard-8` /
us-west2** (`gcloud/env.sh`). 데이터는 이미 버킷에 있으므로 업로드 단계가 없다.

사전 준비: `gcloud` 로그인, 결제 활성 프로젝트, API 활성화
(`aiplatform.googleapis.com`, `artifactregistry.googleapis.com`,
`cloudbuild.googleapis.com`, `storage.googleapis.com`).

```bash
cd gcloud
cp env.example.sh env.sh          # 최초 1회 (PROJECT_ID/REGION/BUCKET 등)
chmod +x *.sh

# 0. (선택) 버킷의 curated 풀이 있는지 사전점검
./check_data.sh

# 1. 모델 컨테이너 빌드/푸시 (Cloud Build — 로컬 Docker 불필요)
./build_and_push.sh

# 2. 학습 잡 제출 (인자는 train.py로 그대로 전달)
#    ⚠️ 제출 전 예상 학습 시간·비용을 먼저 계산해 확인할 것 (CLAUDE.md §3)
./submit_job.sh --epochs=8 --batch-size=8 --grad-accum=2
./submit_job.sh --unfreeze-top=4 --lr=2e-5 --epochs=6

# 진행 확인
gcloud ai custom-jobs list --region=us-west2
```

동작 방식:
- 잡은 GCS 버킷을 `/gcs/<bucket>`으로 FUSE 마운트한다. `submit_job.sh` 가
  `CV_CURATED_ROOT=gs://<bucket>/curated` 를 주입하고, `config.py`가 마운트 경로로
  변환한다. `train.py` 가 `build_splits()` 로 화자분리 split을 컨테이너 안에서 생성.
- `baseOutputDirectory` 아래 `AIP_MODEL_DIR`(= `.../model`)에 모델이 저장된다.
  학습 종료 후 `gs://<bucket>/outputs/classifier/<job>/model/` 에서 산출물 확인.

산출물을 로컬로 내려받아 평가/추론:
```bash
gcloud storage cp -r "gs://qi-ucsd-speech-usw2/outputs/classifier/<job>/model" ./trained_model
cd ../src
python evaluate.py --model-dir ../trained_model
python infer.py clip.mp3 --model-dir ../trained_model
```

옵션·선택 근거(왜 Custom Training인지, GPU 종류 비교 등)는
[`docs/vertex-ai-options.md`](./docs/vertex-ai-options.md) 참고.

---

## 로컬 실행 (스모크 테스트·디버깅용)

로컬 GPU 학습은 **디버깅/스모크 테스트 용도**다. 본 학습은 Vertex(T4)에서 돌린다.
`prepare_data.py` 는 매니페스트(작은 csv)만 읽으므로 로컬에서 가볍게 돌려
분할 구성을 확인할 수 있다. 오디오까지 쓰는 학습은 `CV_CURATED_ROOT` 를
FUSE/로컬 경로로 지정해야 한다.

```bash
# 의존성 (torch는 CUDA에 맞춰 별도 설치 — requirements.txt 상단 주석 참고)
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

cd src

# 분할 매니페스트 구성 확인 (curated manifest만 읽음)
python prepare_data.py --per-class 300

# 학습 (헤드만) → 이후 상단 레이어 unfreeze
python train.py --epochs 8 --batch-size 8 --grad-accum 2
python train.py --unfreeze-top 4 --lr 2e-5 --epochs 6

# 테스트 / 추론
python evaluate.py
python infer.py path/to/clip.mp3
python infer.py clip.mp3 --frames --plot heatmap.png
```

> 하드웨어: 8GB VRAM(RTX 4060 등) 기준 batch 4~8 + fp16, 필요 시 `--grad-accum`,
> `--gradient-checkpointing`.

---

## 로드맵

- **Level 1 (현재):** utterance 단위 억양 퍼센트.
- **Level 2:** `model.py`는 mean pooling 이전 frame-level 로짓(`frame_logits`)을
  이미 보존한다 → 시간축 억양 히트맵(`infer.py --plot`).
- **Level 3 (별도 과제):** 음소 단위 정렬 + 목표 억양 통계 비교로 발음 코칭
  (GOP 계열). read-speech 가정 및 reference 정의 선행 필요.
- **KR 추가:** AI-Hub 데이터 필요(한국 밖 반출 불가 → 서울 리전 버킷). `DATASET.md` §7.
