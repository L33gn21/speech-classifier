# 작업 보고서 (reports/)

> 이 폴더는 classifier 프로젝트의 **작업 로그**다. 지속적 발전이 목표이므로,
> 의미 있는 작업(리팩터링, 인프라 변경, 학습 실험, 데이터셋 재구축 등)을
> 할 때마다 **보고서 한 편**을 남긴다. 나중에 프로젝트를 이어받는 사람(또는
> 에이전트)이 "무엇을 왜 했는지"를 이 폴더만 읽어도 복원할 수 있어야 한다.

> ⚠️ 여기(`classifier/reports/`)는 사람이 읽는 **마크다운 작업 보고서**다.
> GCS 버킷의 `gs://qi-ucsd-speech-usw2/reports/*.json`(큐레이션·통계용
> 기계 판독 리포트)과는 **다른 것**이다. 헷갈리지 말 것.

---

## 언제 쓰나

- 인프라 변경(리전 이동, 리소스 삭제/생성, 비용에 영향 주는 일)
- 데이터셋 재구축·큐레이션 변경
- 학습/평가 실험(하이퍼파라미터, 결과, 비용) — 특히 **의미 있는 결과가 나온 잡**
- 코드 구조 개편, 문서 정비
- 그 외 "다음 사람이 알아야 할" 판단·결정

작은 오타 수정 같은 건 보고서 대상이 아니다. **결정과 결과**가 남을 만한 일에 쓴다.

## 어떻게 쓰나

1. [`TEMPLATE.md`](./TEMPLATE.md) 를 복사한다.
2. 파일명은 **`YYYY-MM-DD-<짧은-슬러그>.md`** (예: `2026-07-16-us-central1-decommission.md`).
   같은 날 여러 건이면 슬러그로 구분한다.
3. 작성 후 아래 **색인**에 한 줄 추가한다(최신이 위로).

## 색인 (최신순)

| 날짜 | 보고서 | 요약 |
|------|--------|------|
| 2026-07-17 | [정식버전 학습 v2 (전체 파인튜닝)](./2026-07-17-production-training-v2-fullft.md) | v1 근거로 전체 파인튜닝(unfreeze-12, lr 3e-5) 채택. 내부 test macro-F1 0.545→0.608, VoxForge acc 0.617→0.704/5클래스 macro-F1 0.625→0.706, 전 클래스↑·CN 과예측 44→6 해소. Model Registry v2. register_model.sh 버그 3건 수정(auto-register 정상화). |
| 2026-07-17 | [정식버전 학습 v1](./2026-07-17-production-training-v1.md) | 헤드-온리→백본 파인튜닝 전환. 클래스가중·early stopping·증강·hypertune 코드 추가, HP 스윕으로 lr 1e-4/unfreeze-4 선정. 내부 test macro-F1 0.437→0.545(+25%), VoxForge 미지 코퍼스 acc 0.617/5클래스 macro-F1 0.625. Model Registry v1 등록. register_model.sh 버그 2건 수정. |
| 2026-07-16 | [curated 풀 5k 재구축](./2026-07-16-curated-5k-rebuild.md) | 클립 수 부족 해소: raw GLOBE 재고를 활용해 큐레이션 캡 상향(화자 100→380/성별, CN 클립캡 해제). curated 풀 8,129→28,237 클립(US/UK/CA ~6k, AU/IN ~4k, CN 1.17k 유지). 기존 풀은 `_archive/curated_v1_8k/`로 보존. TARGET_PER_CLASS 300→5000. |
| 2026-07-16 | [대시보드 정보 확장](./2026-07-16-dashboard-metrics-enrichment.md) | 데이터셋 대시보드에 오디오 용량·평균 크기·추정 길이 추가(메타데이터만), 모델 테스터에 나라별 정확도/F1 막대 + 혼동 행렬 히트맵 추가. train.py가 상세 지표를 final_metrics.json에 저장. |
| 2026-07-16 | [us-central1 정리 & 문서 최신화](./2026-07-16-us-central1-decommission.md) | us-west2 이전 완료에 따라 us-central1 스택(TensorBoard·Artifact·Cloud Run·Custom Job) 철거, 버킷만 보존. 문서 최신화 및 보고서 체계 신설. |

---

각 보고서 상단의 **상태(Status)** 표기 관례:
- `완료(done)` — 작업이 끝나고 검증됨
- `진행중(in-progress)` — 아직 끝나지 않음(예: 모델 학습·테스트 중)
- `보류(blocked)` — 외부 결정/자원 대기
