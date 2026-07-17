# classifier — 운영 규칙 (gcloud-first)

> 이 파일은 `classifier/`에서 작업할 때 **자동으로 로드되는 에이전트 메모리**다.
> 여기 적힌 규칙은 "가능하면"이 아니라 **기본 동작**이다. 데이터·모델은 GCP에서
> 다루고, 로컬은 소스코드·환경변수·문서만 관리한다.
> 데이터셋 상세는 [`DATASET.md`](./DATASET.md), Vertex 옵션은
> [`docs/vertex-ai-options.md`](./docs/vertex-ai-options.md)를 본다.

---

## 0. 핵심 원칙 (먼저 읽을 것)

1. **gcloud-first.** 데이터셋과 모델은 GCS/Vertex에서 관리한다. 로컬 디스크에
   대용량 오디오나 학습 산출물을 만들지 않는다. 로컬의 역할은 **소스코드,
   환경변수(`gcloud/env.sh`), 문서**뿐이다.
2. **데이터 다운로드는 로컬 경유 금지.** "로컬 다운 → 업로드"를 하지 않는다.
   GCP Compute Engine 인스턴스를 띄워 **버킷으로 직접** 받는다 (§2).
   가벼운 메타데이터(매니페스트, csv, parquet 헤더 등)만 로컬에서 받아 분석해도 된다.
3. **학습은 항상 예상 시간·비용을 먼저 보고한다.** T4는 유료다. 잡을 제출하기
   전에 예상 소요 시간과 대략 비용을 계산해서 사용자에게 알린다 (§3).
4. **T4 / us-west2 고정.** 학습 하드웨어와 리전은 아래 좌표를 기본으로 쓴다.
5. **작업마다 보고서를 남긴다.** 지속적 발전이 목표다. 인프라 변경·데이터 재구축·
   의미 있는 학습 실험·구조 개편 등을 하면 `reports/`에 보고서 한 편을 쓴다
   (양식 `reports/TEMPLATE.md`, 색인 `reports/README.md`). 사소한 수정은 제외.

---

## 1. 프로젝트 좌표 (기본값)

| 항목 | 값 | 출처 |
|------|-----|------|
| GCP 프로젝트 | `qi-ucsd-project` | `gcloud/env.sh` |
| 리전 | **us-west2** | `gcloud/env.sh` |
| GCS 버킷 | `gs://qi-ucsd-speech-usw2` | `DATASET.md`, `env.sh` |
| 학습 데이터 | `gs://qi-ucsd-speech-usw2/curated/<CC>/` | `DATASET.md` §1 |
| 모델 산출물 | `gs://qi-ucsd-speech-usw2/outputs/classifier/<JOB>/model/` | `submit_job.sh` |
| GPU | **NVIDIA T4 ×1** | `env.sh` |
| 머신 타입 | `n1-standard-8` | `env.sh` |
| 컨테이너 | `us-west2-docker.pkg.dev/qi-ucsd-project/speech-classifier/accent-classifier:latest` | `env.sh` |
| 학습 클래스 | `US, UK, CA, AU, IN, CN` (6클래스, us-west2 재구축) | `src/config.py` |

- 실제 값은 항상 `gcloud/env.sh`가 소스 오브 트루스다 (gitignore됨 — 커밋 안 함).
- **us-west2 이전 완료.** 옛 us-central1 스택(TensorBoard·Artifact·Cloud Run·Custom
  Job)은 2026-07-16 철거했고 **데이터 버킷 `gs://qi-ucsd-speech-us`만 아카이브로
  보존**한다 (`reports/2026-07-16-us-central1-decommission.md`). 잡 제출 전 **us-west2
  T4 쿼터를 반드시 재확인**한다 — 이전에 확인한 "1개"는 us-central1 기준이었다.
- KR 클래스와 AI-Hub 데이터는 **한국 밖 반출 불가** → us-west2(미국) 버킷에 절대 올리지
  않는다. 필요 시 `asia-northeast3`(서울) 버킷을 따로 쓴다 (`DATASET.md` §7).

---

## 1.5 현재 최고 모델 & 다음 작업 (핸드오프 — 새 세션은 여기부터)

> 이어받는 세션(사람/AI)이 **바로 재현·발전**할 수 있게 남기는 현재 상태.
> 상세 근거는 `reports/`(최신순 색인 `reports/README.md`)를 읽는다.

- **현재 최고 모델: v2 (전체 파인튜닝).** Model Registry `accent-classifier` **v2** (us-central1).
  - 산출물: `gs://qi-ucsd-speech-usc1/outputs/classifier/accent-classifier-20260717-082914-v2final/model/`
  - 성능(내부 화자분리 test): **macro-F1 0.608 / acc 0.620**. 미지 코퍼스 **VoxForge acc 0.704 / 5클래스 macro-F1 0.706**. 클래스별·비교는 `reports/2026-07-17-production-training-v2-fullft.md`.
  - **재현 레시피** (컨테이너 최신 상태에서): 
    ```bash
    cd gcloud && ./build_and_push.sh   # 코드 바꿨으면 먼저 재빌드
    JOB_SUFFIX=v3 ./submit_job.sh --auto-register \
      --unfreeze-top=12 --lr=3e-5 --class-weight=balanced \
      --warmup-ratio=0.15 --per-class=5000 --epochs=10 \
      --early-stopping-patience=3 --augment
    ```
- **⚠️ 현재 컴퓨트/데이터는 임시로 us-central1**(env.sh: `REGION=us-central1`,
  `BUCKET=qi-ucsd-speech-usc1`). us-west2 T4 재고 복귀 시 되돌린다. §1 표는 원 홈(us-west2)
  기준이니 **env.sh가 소스 오브 트루스**.
- **학습 방법론(검증됨):** ① 짧은 프록시 스윕(`sweep.sh`, `--per-class 1500 --epochs 3~4`)으로
  구조(unfreeze/lr/class-weight) 결정 → ② 승자로 전량 정식 학습. 전체 파인튜닝(uf12)이
  부분 unfreeze(uf4)를 확실히 상회했고 소수 클래스(CN)도 더 잘 잡았다.
- **다음 개선 후보 (우선순위):**
  1. **CN recall↑** — 현재 정밀 높고 재현 낮음. 근본책은 SpeechOcean762(본토 만다린) raw
     인제스트로 CN 데이터 확충(`DATASET.md` §5). 임시로 `--class-weight sqrt`/임계값 실험.
  2. **US↔CA 혼동** 완화(북미권) — 추가 데이터/피처.
  3. **HP 정밀 탐색** — `submit_hp_tuning_job.sh`(Vertex Vizier, `train.py --hypertune`
     연동 완료)로 weight_decay·warmup·dropout·lr 자동 최적화(비용=trial수×학습).
- **튜닝 가능한 인자**는 `train.py` argparse에 노출: `--lr --unfreeze-top --class-weight
  {none,balanced,sqrt} --dropout --warmup-ratio --weight-decay --epochs
  --early-stopping-patience --augment --per-class`. 상세는 `docs/hyperparameter-tuning.md`.

---

## 2. 데이터셋 다운로드 규칙 (VM → 버킷 직접)

새 데이터 소스를 버킷에 추가할 때:

1. 먼저 **메타데이터만 로컬**로 받아 분석한다 (클래스 분포, 라이선스, 스키마).
   무엇을 받을지 확정되면,
2. **us-west2에 Compute Engine 인스턴스를 띄워** 그 안에서 다운로드하고
   `gsutil`/`gcloud storage`로 곧장 `gs://qi-ucsd-speech-usw2/...`에 쓴다.
   로컬 디스크를 경유하지 않는다 (수십 GB를 집 네트워크로 왕복시키지 않기 위함).
3. 작업이 끝나면 **인스턴스를 반드시 삭제**한다 (유휴 VM = 낭비 비용).
4. 큐레이션 결과는 `curated/<CC>/manifest.csv` 스키마(`DATASET.md` §2)를 지킨다.
   `curated/`·`raw/`의 스토리지 클래스와 비용은 `DATASET.md` §8 참고
   (`raw/`는 Coldline — 함부로 스캔하면 retrieval 요금 발생).

> 데이터 **업로드 단계는 없다.** 데이터는 이미 버킷 `curated/`에 있고,
> `train.py`가 잡 시작 시 `build_splits()`로 화자분리 split을 컨테이너 안에서
> 직접 만든다. 제출 전 버킷에 풀이 있는지 `gcloud/check_data.sh`로 사전점검한다.
> (옛 `upload_data.sh`는 제거됨.)

---

## 3. 모델 학습 규칙 (T4 / us-west2, 비용 보고 필수)

학습은 Vertex AI Custom Job으로 돌린다. 로컬 GPU 학습은 디버깅·스모크 테스트용으로만.

**잡 제출 전 반드시 예상 시간·비용을 계산해 보고한다:**

```
steps  = ceil(총_클립수 / (batch_size × grad_accum)) × epochs
ETA    ≈ steps / (T4 처리량 step/s)      # 실측 없으면 첫 로그의 it/s로 보정
비용   ≈ ETA(시간) × T4 커스텀학습 시간당 단가(대략 $0.4~0.5/h 수준, 리전·요금제 확인)
```

- 대략 감(anchor, 실측으로 갱신할 것): `TARGET_PER_CLASS=300` 4클래스
  프로토타입(~1.2k 클립)은 T4에서 수십 분 내. 큐레이션 풀 전량(~2만 클립)을 다중
  epoch 돌리면 수 시간대. **정확한 값은 첫 스텝 로그의 it/s로 다시 계산해 보고.**
- 잡 종료 후 **결과와 실제 소요 시간·비용을 다시 보고**한다.

표준 워크플로 (`gcloud/` 안에서):

```bash
cd gcloud
cp env.example.sh env.sh        # 최초 1회 (값은 이미 채워져 있음, gitignore됨)

./build_and_push.sh             # ② 컨테이너 빌드 → Artifact Registry (Cloud Build)
./submit_job.sh --epochs=4 --batch-size=8 --grad-accum=2   # ③ Custom Job 제출
                                #    인자는 그대로 train.py로 전달됨

gcloud ai custom-jobs list --region=us-west2            # 진행 추적
```

- 데이터 소스는 `submit_job.sh`가 `CV_CURATED_ROOT=gs://.../curated`로 주입한다.
  `config.py`가 `gs://` → `/gcs/` FUSE 경로로 자동 변환한다.
- `--auto-register`를 붙이면(`./submit_job.sh --auto-register --epochs=8`) 잡 완료까지
  대기했다가 성공 시 `register_model.sh`로 Model Registry에 자동 등록한다. 완료까지
  블록되므로(수 시간 가능) 백그라운드(`&`)로 돌려도 된다. 실패/취소면 등록하지 않는다.
- 산출물 확인/다운로드:
  ```bash
  gcloud storage cp -r "gs://qi-ucsd-speech-usw2/outputs/classifier/<JOB>/model" ./trained_model
  ```

### 3.1 학습 시각화 (Vertex AI TensorBoard)

`env.sh`에 `TENSORBOARD_ID`/`SERVICE_ACCOUNT`가 채워져 있으면 `submit_job.sh`가
Custom Job에 자동으로 TensorBoard를 연결한다. `train.py`는
`AIP_TENSORBOARD_LOG_DIR`(Vertex가 주입하는 GCS 경로)에 TensorBoard 로그를
쓰고, Vertex가 이를 학습 도중 실시간으로 인스턴스에 동기화한다. loss·accuracy·
macro F1·클래스별 F1 커브를 GCP 콘솔에서 바로 볼 수 있다:

```
https://console.cloud.google.com/vertex-ai/experiments/tensorboard-instances?project=qi-ucsd-project
```

- 인스턴스는 최초 1회만 만들면 된다: `gcloud ai tensorboards create --display-name=speech-classifier --region=us-west2`
- `TENSORBOARD_ID`/`SERVICE_ACCOUNT`가 비어 있으면 `submit_job.sh`는 TensorBoard
  연결 없이 평소대로 제출한다(선택 기능).
- 로컬(비-Vertex) 실행 시에는 `outputs/classifier/tb_logs`에 로그가 쌓이며,
  `tensorboard --logdir outputs/classifier/tb_logs`로 로컬에서 볼 수 있다.

---

## 4. 로컬 vs 클라우드 책임 분리

| 항목 | 어디서 | 비고 |
|------|--------|------|
| 소스코드 (`src/`, `gcloud/`) | **로컬** (git) | 로컬/Vertex 동일 코드 |
| 환경변수 (`gcloud/env.sh`) | **로컬** | gitignore, 커밋 금지 |
| 문서 (`*.md`, `docs/`) | **로컬** (git) | 이 파일 포함 |
| 데이터셋 (오디오·매니페스트) | **GCS** `curated/`,`raw/` | 로컬로 대량 복사 금지 |
| 대용량 다운로드 | **Compute Engine → 버킷** | 로컬 경유 금지 (§2) |
| 학습 | **Vertex Custom Job (T4)** | 로컬은 스모크 테스트만 |
| 모델 산출물 | **GCS** `outputs/classifier/` | 필요 시 로컬로 내려받아 평가 |
| 메타데이터 분석 | 로컬 OK | 가벼운 csv/parquet 헤더 정도 |

---

## 5. 파일 맵

| 경로 | 역할 |
|------|------|
| `DATASET.md` | 버킷 레이아웃·클래스·큐레이션·캐비엇 (데이터 소스 오브 트루스) |
| `reports/` | 작업 보고서(작업 로그) — 작업마다 한 편, `TEMPLATE.md`/색인 `README.md` |
| `docs/vertex-ai-options.md` | Vertex 서비스/옵션 선택 근거 (Custom Training 확정) |
| `src/config.py` | 경로·라벨·하이퍼파라미터 (env로 로컬/GCS 겸용) |
| `src/{prepare_data,dataset,model,train,evaluate,infer}.py` | 학습·추론 로직 (로컬/Vertex 공통) |
| `Dockerfile` | 학습 컨테이너 (CUDA torch + ffmpeg) |
| `gcloud/env.sh` | 프로젝트·버킷·머신 값 (gitignore) |
| `gcloud/build_and_push.sh` | 이미지 빌드 → Artifact Registry |
| `gcloud/submit_job.sh` | Vertex Custom Job 제출 (T4) |
| `gcloud/register_model.sh` | 학습 산출물(GCS) → Model Registry 등록 (카탈로그/버전관리) |
| `gcloud/check_data.sh` | 버킷 curated 풀 사전점검 (업로드 단계는 없음, §2) |
