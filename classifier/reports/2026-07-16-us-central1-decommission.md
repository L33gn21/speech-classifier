# us-central1 스택 철거 & 문서 최신화

- **날짜:** 2026-07-16
- **작성:** Claude (에이전트)
- **상태:** 완료(done) — 단, 모델 학습·테스트는 별도로 진행중(in-progress)
- **범위:** 인프라 / 문서

## 요약 (TL;DR)

classifier가 us-west2로 완전 이전 완료됨에 따라, 기존 **us-central1** 리전의
Vertex/컨테이너/서비스 리소스를 모두 철거했다. **데이터 버킷
`gs://qi-ucsd-speech-us`는 아카이브로 보존**한다(지시). 이어서 classifier 문서를
us-west2·6클래스 현행 상태로 최신화하고, 지속적 발전을 위한 **작업 보고서 체계
(`classifier/reports/`)** 를 신설했다.

## 배경 / 동기

- 프로젝트가 us-central1 → **us-west2**로 이전 완료(신 버킷 `gs://qi-ucsd-speech-usw2`,
  신 Artifact/Cloud Run/코드 모두 us-west2에 존재함을 확인).
- us-central1에는 이전 과정의 잔여 리소스가 남아 스토리지·TensorBoard 과금과
  혼동의 소지가 있었다.
- 일부 문서가 옛 상태(이전 중 표현, 구 클래스 라벨)로 남아 있었다.

## 수행 작업

### 1. us-central1 리소스 철거 (버킷 제외 전부)

이전에 us-west2로 이미 옮겨졌음을 확인한 뒤 삭제:

| 리소스 | 상세 | 조치 |
|--------|------|------|
| Vertex AI TensorBoard `speech-classifier` | `.../us-central1/tensorboards/1238615241852452864` | 삭제 |
| Artifact Registry `speech-classifier` | 이미지 7개 ≈ 25 GB | 삭제 |
| Artifact Registry `cloud-run-source-deploy` | Cloud Run 소스 이미지 | 삭제 |
| Cloud Run `accent-dataset-dashboard` (us-central1) | us-west2에 라이브 사본 존재 | 삭제 |
| Cloud Run `accent-model-tester` (us-central1) | us-west2에 라이브 사본 존재 | 삭제 |
| Vertex Custom Job 11개 | 전부 종료 상태(SUCCEEDED/FAILED/CANCELLED) | 삭제 |
| Compute Engine 인스턴스 | 없음(이미 정리됨) | 조치 없음 |

> Custom Job은 `gcloud ai custom-jobs`에 `delete` 서브커맨드가 없어
> REST `DELETE .../customJobs/<id>`로 삭제함(삭제 반영이 리스트에 지연 반영되어
> 반복 확인함).

**보존:** 데이터 버킷 `gs://qi-ucsd-speech-us`(US-CENTRAL1) — 삭제하지 않음.
`run-sources-qi-ucsd-project-us-central1` 스테이징 버킷도 버킷이라 보존.

### 2. 철거 전 학습 이력 기록 (us-central1 Custom Job)

삭제 전에 남긴 스냅샷(2026-07-15 실험 세션):

| Job | 상태 | 소요(대략) |
|-----|------|-----------|
| accent-classifier-20260715-142831 | SUCCEEDED | ~22분 |
| accent-classifier-20260715-141407 | FAILED | ~6분 |
| accent-classifier-20260715-133744 | SUCCEEDED | ~22분 |
| accent-classifier-20260715-105527 | SUCCEEDED | ~14분 |
| accent-classifier-20260715-104210 | SUCCEEDED | ~10분 |
| accent-classifier-20260715-103310 | SUCCEEDED | ~7분 |
| accent-classifier-20260715-101459 | CANCELLED | — |
| accent-classifier-20260715-101127 | CANCELLED | — |
| accent-classifier-20260715-083920 | SUCCEEDED | ~31분 |
| accent-classifier-20260715-082012 | CANCELLED | — |
| accent-classifier-20260715-075151 | FAILED | ~17분 |

(us-central1은 옛 데이터셋 스코프였다. us-west2 6클래스 학습은 새로 돌린다.)

### 3. 문서 최신화

- `classifier/README.md` — 학습 클래스 표기를 옛 4클래스(`US/UK/IN/NG`)에서
  실제 **6클래스(`US/UK/CA/AU/IN/CN`)** 로 정정(`config.py`·`DATASET.md`와 일치).
- `classifier/CLAUDE.md` — "us-west2로 옮기는 중" → **이전 완료** 표현으로 갱신,
  us-central1 쿼터/TensorBoard 관련 잔재 정리. `reports/` 규칙·파일맵 추가.
- `classifier/gcloud/env.sh` — 삭제된 us-central1 TensorBoard 인스턴스 참조 주석
  정리(재생성 안내만 유지).

### 4. 보고서 체계 신설

- `classifier/reports/` 생성: `README.md`(색인·작성법), `TEMPLATE.md`(양식),
  그리고 본 보고서.

## 결과 / 검증

철거 후 us-central1 리소스 카운트(모두 0 확인):

```
TensorBoards(us-central1) : 0
Custom Jobs(us-central1)  : 0
Artifact repos(us-central1): 0
Cloud Run(us-central1)    : 0
Bucket gs://qi-ucsd-speech-us : 보존됨 (location US-CENTRAL1)
```

## 비용 / 영향

- **절감:** us-central1 Vertex TensorBoard 인스턴스(월 과금) + Artifact Registry
  이미지 ~25 GB 스토리지 제거.
- **보존 비용:** 아카이브 버킷 `gs://qi-ucsd-speech-us`의 스토리지 비용은 유지.
  (필요 없어지면 별도 판단으로 정리 가능 — 현재는 지시대로 보존.)
- 실행중 서비스 중단 영향 없음(us-west2 사본이 대체).

## 후속 조치 (TODO)

- [ ] us-west2 T4 쿼터 확인 후 6클래스 본 학습 제출(진행중).
- [ ] us-west2에 Vertex TensorBoard 재생성 → `gcloud/env.sh`의 `TENSORBOARD_ID` 채우기.
- [x] 레포 루트의 **옛 us-central1 문서** `DATASET.md`·`TESTSET.md` 삭제(사용자 결정).
      옛 프로젝트 잔재(`src/`, `언어-감지기-(Language-Detector)/`, `test_samples/`)는 유지.
- [ ] 아카이브 버킷 `gs://qi-ucsd-speech-us`의 `raw/`를 Coldline로 전환 검토(스토리지 절감).

## 참고

- 신 좌표: `classifier/CLAUDE.md` §1, `classifier/gcloud/env.sh`
- 데이터셋: `classifier/DATASET.md`(us-west2 rebuild)
