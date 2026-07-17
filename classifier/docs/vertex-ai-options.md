# Vertex AI 학습 옵션 가이드 (accent classifier)

> 이 프로젝트(`classifier/`)를 GCP **Vertex AI Custom Training**으로 학습할 때
> 각 단계에서 어떤 서비스/옵션을 선택할 수 있는지 정리한 문서.
> 결론부터: **기존 파이썬 학습 코드는 그대로 재사용**되고, Vertex는 그 코드를
> GPU 위에서 컨테이너로 실행해주는 실행 환경일 뿐이다.

---

## 0. 핵심 전제

- Vertex AI Custom Training = "내 학습 코드를 대신하는 것"이 **아니라**,
  "내 파이썬 코드(`train.py` 등)를 GCP GPU 위에서 컨테이너로 돌려주는 실행 환경".
- `train.py`, `model.py`, `dataset.py`, `config.py` 등 학습 로직은 **한 벌만 유지**되고
  로컬/Vertex 양쪽에서 동일하게 동작한다.
  - `config.py`의 `gcs_to_fuse()` → `gs://버킷/...`를 Vertex 마운트 경로 `/gcs/...`로 자동 변환
  - `OUTPUT_DIR`는 `AIP_MODEL_DIR`(Vertex 주입) → 없으면 로컬 `outputs/`로 폴백
  - 실행은 로컬이든 컨테이너든 `python train.py --epochs 8` 동일

---

## 1. 모델을 "만드는 방식"이 3갈래

| 접근 | 무엇인가 | 코드 필요? | 언제 |
|---|---|---|---|
| **AutoML** | 데이터만 올리면 구글이 모델 구조·학습을 알아서 | ❌ 없음 | 이미지/표/텍스트/영상, 표준 문제 |
| **Custom Training** | **내 학습 코드(train.py)를 컨테이너로 실행** | ✅ 내 코드 | 커스텀 모델, 특수 도메인 |
| **Model Garden / 사전학습** | Gemini, Llama, 공개 모델을 그대로/파인튜닝 | 대개 ❌ | 파운데이션 모델 활용 |

> ⚠️ **이 프로젝트에서 AutoML은 탈락.**
> AutoML은 이미지·표·텍스트·영상만 지원하고 **원시 오디오 분류를 지원하지 않는다.**
> 이 프로젝트는 wav2vec2로 오디오 파형을 직접 다루는 HuggingFace 커스텀 모델이라
> **Custom Training이 정답**이며, 그게 이미 `classifier/`에 세팅된 방식이다.
> (편법으로 오디오를 스펙트로그램 이미지로 변환해 이미지 AutoML에 넣을 수는 있으나,
>  지금 코드를 버리게 되므로 무의미.)

**→ 모델 방식은 Custom Training으로 확정.**

---

## 2. 단계별 선택지

### ① 데이터 저장 — 어디에 둘 것인가
- **Cloud Storage (GCS 버킷)** ← 채택. curated 오디오/매니페스트가 버킷의
  `curated/`에 있고(VM→버킷으로 빌드, `CLAUDE.md` §2), Vertex가 `/gcs/`로
  마운트해서 읽음. 오디오처럼 큰 파일엔 표준. **로컬 업로드 단계는 없다** —
  제출 전 `gcloud/check_data.sh`로 풀 존재만 사전점검.
- (선택) **Vertex "Managed Dataset"** — AutoML용 라벨링/분할 관리 도구.
  Custom Training엔 **불필요**.

**→ 선택: GCS 버킷 그대로. 추가 결정 없음.**

### ② 학습 코드를 담는 방식 — 컨테이너
- **Custom Container (내 Dockerfile)** ← 채택. torch+ffmpeg+내 코드 통째로.
  재현성 최고. (`Dockerfile` + `gcloud/build_and_push.sh`)
- **Pre-built Container + 코드만 전달** — 구글 제공 torch 이미지에 스크립트만 얹기.
  간단하지만 ffmpeg 같은 시스템 의존성 넣기 번거로움.

**→ 선택: Custom Container 그대로.** (오디오 디코딩에 ffmpeg 필요하므로 이게 정답)

### ③ 학습 실행 형태 — Job 종류
| 옵션 | 설명 | 이 프로젝트 |
|---|---|---|
| **Custom Job** | 컨테이너 1회 실행. 가장 기본 | ✅ **채택** (`submit_job.sh` → `custom-jobs create`) |
| Hyperparameter Tuning Job | lr·batch 등 자동 탐색 | 나중에 튜닝 시 선택적 |
| Training Pipeline | 데이터셋+학습+모델등록 묶음 | AutoML/파이프라인용, 불필요 |

**→ 선택: Custom Job.** 나중에 lr/unfreeze 조합 자동 탐색이 필요하면
HP Tuning Job으로 승격 가능(코드 거의 그대로).

### ④ 컴퓨팅 — 머신 & GPU (실제 비용/속도 결정)
`gcloud/env.example.sh`에서 고르는 부분:

| GPU | 머신 타입 | 특징 | 비용감 |
|---|---|---|---|
| **T4** | `n1-standard-8` | 싸고 무난, wav2vec2-base 파인튜닝에 충분 | 가장 저렴 |
| **L4** | `g2-standard-*` | T4보다 빠르고 최신, fp16 좋음 | 중간 |
| **A100/H100** | `a2-*` / `a3-*` | 대형 모델·대량 데이터용 | 비쌈 (현재 규모엔 과함) |

**→ 선택: T4로 시작 (현재 기본값).** 느리면 L4로. A100은 지금 규모엔 과투자.

### ⑤ 학습 관리·부가 기능 (선택, 나중에)
- **Vertex Experiments** — 실험별 metric 추적/비교.
  현재 `report_to="none"`이라 꺼져 있음. 켜면 여러 학습 비교 편함.
- **TensorBoard (Vertex 호스팅)** — 학습 곡선 시각화
- **HP Tuning** — ③ 참고

**→ 전부 옵션. MVP 돌린 뒤 붙이면 됨.**

### ⑥ 학습 후 — 모델을 어떻게 쓸까
| 옵션 | 설명 | 이 프로젝트 |
|---|---|---|
| GCS에 산출물만 저장 | 모델 파일을 버킷에 두고 필요 시 다운로드 | ✅ 현재 `AIP_MODEL_DIR`가 이걸 함 |
| **Model Registry 등록** | 버전 관리 | 선택 — `gcloud/register_model.sh` (카탈로그/버전관리용). 이 커스텀 모델은 prebuilt 서빙 컨테이너로 배포 불가라, 실제 서빙엔 infer.py를 감싼 커스텀 서빙 컨테이너가 별도로 필요 |
| **Endpoint 배포** | 상시 REST API로 실시간 추론 | 서빙 필요할 때 |
| **Batch Prediction** | 대량 파일 한 번에 추론 | 오프라인 대량 처리 시 |

**→ 현재 목표는 GCS 저장까지. 서빙은 별도 단계.**

---

## 3. 최종 조합 요약

```
데이터   : Cloud Storage 버킷 (curated/)  ✅ 확정 (업로드 단계 없음, check_data.sh로 점검)
모델방식 : Custom Training                ✅ 확정 (AutoML은 오디오 미지원)
컨테이너 : Custom Container (Dockerfile)  ✅ 확정 (ffmpeg 필요)
잡 종류  : Custom Job                     ✅ 확정 (submit_job.sh)
컴퓨팅   : T4 GPU / n1-standard-8         ← 여기만 취향껏 조정
── 이후 선택(나중에) ──
튜닝     : Hyperparameter Tuning Job      (원하면)
추적     : Vertex Experiments / TensorBoard (원하면)
서빙     : Endpoint or Batch Prediction   (필요하면)
```

**핵심:** AutoML·Managed Dataset 등은 이 시나리오에선 안 써도 된다.
`classifier/`는 이미 가장 정석적인 Custom Training 파이프라인으로 짜여 있고,
실제로 "선택"할 여지가 있는 건 사실상 **④ GPU 종류**와 **⑤⑥ 나중에 붙일 부가 기능**뿐이다.

---

## 4. 관련 파일 맵

| 파일 | 역할 | Vertex 전환 시 |
|---|---|---|
| `config.py` `model.py` `dataset.py` `train.py` `evaluate.py` `infer.py` | 순수 학습/추론 로직 | **그대로 재사용** |
| `Dockerfile` | 학습 코드를 담는 컨테이너 (CUDA torch + ffmpeg) | Vertex가 이 이미지를 실행 |
| `gcloud/build_and_push.sh` | 이미지 빌드 → Artifact Registry | Cloud Build 사용 |
| `gcloud/check_data.sh` | 버킷 curated 풀 사전점검 | 업로드 아님(데이터는 이미 버킷) |
| `gcloud/submit_job.sh` | Custom Job 제출 | `gcloud ai custom-jobs create` |
| `gcloud/env.example.sh` | 프로젝트/버킷/머신 설정 | `env.sh`로 복사해 값 채움 |

## 5. 실행 순서 (요약)

1. `gcloud/env.example.sh` → `gcloud/env.sh`로 복사 후 프로젝트/버킷 값 채우기
2. `gcloud/check_data.sh` — 버킷 `curated/` 풀 존재/클립수 사전점검 (업로드 아님)
3. `gcloud/build_and_push.sh` — Cloud Build로 이미지 빌드 → Artifact Registry 푸시
4. `gcloud/submit_job.sh --epochs=8 --batch-size=8 --grad-accum=2` — Custom Job 제출
   (제출 전 예상 학습 시간·비용을 먼저 계산해 확인 — `CLAUDE.md` §3)
5. `gcloud ai custom-jobs list --region=<REGION>` 으로 진행 상황 추적
6. 학습 산출물은 `gs://<BUCKET>/outputs/classifier/<JOB_NAME>/model` 에 저장됨
