# 대시보드 정보 확장 — 데이터셋 용량/길이 + 모델 나라별 성능 시각화

- **날짜:** 2026-07-16
- **작성:** Claude (에이전트)
- **상태:** 완료(done) — 코드/로컬 검증 완료, Cloud Run 재배포는 후속
- **범위:** 코드 (대시보드 2종 + 학습 스크립트)

## 요약 (TL;DR)

두 대시보드가 너무 적은 정보만 보여준다는 요청에 따라 정보량을 늘렸다.
- **데이터셋 대시보드:** 클립 수만 나오던 것에 **오디오 총 용량·평균 클립 크기·추정
  총 길이**를 추가했다. 용량은 GCS 오브젝트 메타데이터(`blob.size`)만 읽어 정확히
  집계하며 오디오 자체는 내려받지 않는다(curated=Standard, 비용 부담 없음).
- **모델 테스터:** 정확도 하나만 나오던 것에 **나라별 정확도/F1 막대 + 혼동 행렬
  히트맵 + 통계 카드**를 추가했다. 나라별 F1은 이미 `final_metrics.json`에 저장돼
  있었으나 화면에서 버려지고 있었고, 혼동 행렬·정밀도/재현율은 `train.py`가 새로
  저장하도록 했다.

## 배경 / 동기

- 데이터셋 대시보드(`serve_dataset_report.py`)는 클래스별 클립 수·화자 수·성비·출처만
  보여줬다. 사용자가 "파일 용량이나 총합 길이 등"도 보고 싶어 했다.
- 모델 테스터(`serve_model_tester.py`)는 드롭다운에 전체 정확도만 노출했다. 사용자가
  "각 나라별 정확도" 같은 상세치를 시각화로 보고 싶어 했다.

## 수행 작업

### 1. 데이터셋 대시보드 (`src/dashboard/inspect_dataset.py`, `serve_dataset_report.py`)
- `collect_audio_stats(root, classes)` 추가: `curated/<CC>/audio/*` 오브젝트를
  나열해 `blob.size`를 합산하고 fname 접두어(`glb_`/`saa_`)로 소스별로 쪼갠다.
  **메타데이터만** 읽는다(다운로드 없음). 로컬 경로도 지원(`stat().st_size`).
- 요약표에 **Size / Avg clip / Est. dur.** 컬럼과 **Total 합계 행** 추가.
- 상단에 **통계 카드**(Classes/Clips/Speakers/Total size/Avg clip/Est. duration) 추가.
- **Storage per class by source** 누적 막대 차트 추가.
- 길이(duration)는 manifest에 컬럼이 없어 정확 계산 불가 → **파일 크기 기반
  코덱별 추정치**로 제공하고 화면에 "est." + 설명 각주로 명시(오해 방지).
- 용량 집계 실패(권한 등)는 조용히 건너뛰어 기존 클립수 화면을 절대 깨지 않게 함.

### 2. 학습 스크립트 (`src/train.py`)
- `detailed_report(preds, labels)` 추가: 클래스별 precision/recall/f1/support +
  혼동 행렬 계산.
- 최종 평가를 `trainer.evaluate` → `trainer.predict`로 바꿔 원본 예측을 확보하고,
  `final_metrics.json`에 `eval_detail`/`test_detail`(labels·per_class·confusion_matrix)를
  중첩 저장. 기존 평면 키(`test_accuracy`, `test_f1_US` 등)는 그대로 유지 → 하위 호환.

### 3. 모델 테스터 (`src/model_tester/serve_model_tester.py`)
- `list_models()`가 `has_detail` 플래그를 노출.
- `get_metrics(job)` + `GET /metrics/<job>` 엔드포인트 추가: 선택 모델의 상세 지표를
  반환. **신형 모델**은 `test_detail`을 그대로, **구형 모델**은 평면 `test_f1_*`
  스칼라로 나라별 F1 뷰를 합성(혼동 행렬은 신형만).
- 프론트엔드(순수 HTML/CSS/JS, 외부 라이브러리 없음): 모델 선택 시
  통계 카드 + 나라별 recall/F1 막대(정밀도·support 부기) + 행 정규화 혼동 행렬
  히트맵(대각선 강조)을 렌더링.

## 결과 / 검증

- **데이터셋 대시보드:** 로컬 합성 픽스처(US 3클립/CN 2클립, glb_/saa_ 혼합)로
  `inspect_dataset.py` CLI 실행 → 통계 카드·용량 컬럼·합계 행·추정 길이·5개 차트가
  정상 생성됨(총 285.2 KB, 평균 57 KB, ≈15s 표시 확인).
- **모델 테스터:** `PAGE.format()`이 중괄호 이스케이프 오류 없이 렌더(len 12,633),
  추출한 `<script>` 본문을 `node --check`로 문법 검증 통과. `get_metrics`의 신형/구형
  분기 로직 검토 완료.
- **train.py:** `detailed_report`는 표준 sklearn 사용(혼동 행렬 행 합 = support).
  변경 4개 파일 전부 `python -m py_compile` 통과.

## 비용 / 영향

- **거의 없음.** 데이터셋 대시보드의 용량 집계는 오브젝트 **나열(list)** + 메타데이터
  읽기뿐 — 오디오 다운로드 없음. curated/는 Standard 스토리지라 retrieval 요금 없음.
- 학습 산출물 스키마가 커졌으나(`final_metrics.json`에 detail 추가) 기존 키는 유지 →
  기존 모델·소비자 코드 영향 없음.

## 후속 조치 (TODO)

- [ ] 두 Cloud Run 서비스 재배포(`src/dashboard/deploy.sh`, `src/model_tester/deploy.sh`).
- [ ] 다음 학습 잡부터 혼동 행렬이 `final_metrics.json`에 포함됨 → 테스터에서 히트맵 확인.
- [ ] (선택) 정확한 오디오 길이가 필요하면 curation 시 duration 컬럼을 manifest에
      기록하거나, 헤더 range-read로 산출하는 옵션 추가 검토(현재는 크기 기반 추정).

## 참고

- 변경 파일: `src/dashboard/inspect_dataset.py`, `src/dashboard/serve_dataset_report.py`,
  `src/train.py`, `src/model_tester/serve_model_tester.py`
- 데이터 스키마: `DATASET.md` §2(manifest에 duration 컬럼 없음), §8(스토리지 클래스)
