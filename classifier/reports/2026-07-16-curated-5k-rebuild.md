# curated 풀 5k-scale 재구축 (US/UK/CA/AU/IN ~5~6k, CN 유지)

- **날짜:** 2026-07-16
- **작성:** Claude (Opus 4.8) + L33gn21
- **상태:** 완료(done)
- **범위:** 데이터셋 / 코드 / 문서

## 요약 (TL;DR)

curated 풀의 클립 수가 너무 적어서(클래스당 ~1.5k, 총 8,129) `raw/`의 여유
재고를 조사했다. GLOBE raw에 581,725 클립(전부 Standard 스토리지)이 있어
5개 클래스는 5k+로 늘릴 여지가 충분했다. GCE VM(us-west2)에서 큐레이션 캡을
올려(GLOBE 100→380 화자/성별, 15→20 클립/화자, CN 클립캡 해제) 재구축했고,
curated 풀을 **8,129 → 28,237 클립**으로 확대했다. CN은 소스 한계(홍콩영어 90
화자)로 ~1.17k에 고정 — 불균형은 학습의 macro-F1 + class weighting으로 흡수.

## 배경 / 동기

- 기존 curated 풀(8.1k): US/UK/CA/AU/IN ~1.5k, CN 679. 화자 수는 준수했으나
  클립 수가 학습에 부족하다고 판단.
- 목표: 클립을 "5천대"로. 화자 수는 늘려도 무방(사용자 승인).
- 조사 결과 GLOBE 화자당 클립이 심하게 우편향(랜덤 샘플 ~7~8클립/화자)이라
  **클립을 늘리려면 클립/화자가 아니라 화자 수를 늘려야** 했다. raw 재고는 충분:
  US 15,948 · UK 3,490 · CA 910 · AU 833 · IN 540 화자(GLOBE).
- CN 병목: GLOBE에 본토 만다린 억양 태그가 없어 홍콩영어(90화자/1,096클립)로만
  구성 → 5k 도달 불가. 사용자 결정: **CN은 상한(~1.2k) 유지, 나머지 5k**.

## 수행 작업

1. **로컬 코드/스펙**
   - `spec_5k.json` 신규: `globe_speakers_per_gender: 380`,
     `globe_clips_per_speaker: 20`, CN에 `globe_clips_per_speaker: 200`(사실상
     해제). SAA 40+40 유지, 하이브리드 라벨축 유지.
   - `curate.py`: `--dry-run`(GLOBE pass-1만 실행, 클래스별 화자/클립 수 출력
     후 종료) 및 클래스별 캡 오버라이드 지원 추가.
   - `src/config.py`: `TARGET_PER_CLASS 300 → 5000`.
2. **VM 재큐레이션 (gcloud-first, 로컬 경유 없음)**
   - GCE `curate-5k`(us-west2-a, e2-standard-4, 120GB) 기동.
   - GLOBE 파케이 110샤드(~45GB)를 버킷→VM 병렬 스테이징(동일 리전, egress 무료,
     180MiB/s, 4분).
   - `--dry-run`으로 캡 튜닝(380→320→380 반복, pass-1이 ~5초라 저렴).
   - 전체 `curate.py` 실행 → 로컬(VM) `~/work/curated` 생성(28,237 클립, 2.6GB).
3. **스테이징 → 검증 → 스왑 (전부 버킷 내 서버사이드)**
   - VM→ `gs://qi-ucsd-speech-usw2/curated_5k/` 업로드.
   - 클래스별 audio 객체 수 == manifest 행 수 검증(6/6 OK).
   - 기존 풀 아카이브: `curated/` → `_archive/curated_v1_8k/`(8,135 객체).
   - 승격: `curated_5k/` → `curated/`.
   - VM 삭제.
4. **문서**: `DATASET.md` §1/§3/§4 갱신, 본 보고서 작성.

## 결과 / 검증

재구축 후 클래스별(= manifest 행 수 == audio 객체 수, 검증 완료):

| 클래스 | 클립 | 화자 | F/M/U | 소스 |
|---|---:|---:|---|---|
| US | 6,243 | 840 | 3316/2927/0 | GLOBE 6,163 · SAA 80 |
| UK | 5,933 | 824 | 2740/3193/0 | GLOBE 5,869 · SAA 64 |
| CA | 6,002 | 695 | 2145/3585/272 | GLOBE 5,948 · SAA 54 |
| AU | 4,825 | 642 | 1653/3058/114 | GLOBE 4,792 · SAA 33 |
| IN | 4,064 | 609 | 1183/2881/0 | GLOBE 3,990 · SAA 74 |
| CN | 1,170 | 164 | 349/813/8 | GLOBE 1,096 · SAA 74 |

- 총 **28,237 클립**(기존 8,129 대비 3.5배), 화자 ~250 → 600~840/클래스.
- 최대 클래스 US 6,243 — "1만 이상 급증 금지" 요건 충족.
- IN/AU는 GLOBE 화자 상한(535/609)으로 ~4k에 안착(사용자 허용 범위).
- CN은 홍콩영어 전량 소진(1,096) + SAA로 1,170 — 소스상 이것이 상한.

## 비용 / 영향

- GCE `curate-5k`: e2-standard-4 + 120GB, 총 가동 ~1시간 → **~$1 미만**.
  작업 후 인스턴스 삭제(유휴 과금 없음).
- 데이터 전송: GLOBE 스테이징·업로드·스왑 모두 us-west2 리전 내 → **egress 무료**.
- 스토리지: curated/ 2.6GB(Standard) + `_archive/curated_v1_8k/` 기존 풀 보존
  (재구축 가능하므로 원하면 삭제해 비용 절감 가능).

## 후속 조치 (TODO)

- [ ] 새 풀로 학습 잡 제출 전 T4 쿼터 재확인 후 `submit_job.sh` (비용 보고 필수).
- [ ] `_archive/curated_v1_8k/`는 불필요 판단 시 삭제(스토리지 절감).
- [ ] CN을 진짜로 키우려면 SpeechOcean762(본토 만다린) raw 인제스트 필요(별도 작업).
- [ ] 클립 증가로 채널 누수(GLOBE vs SAA) 프로브 재실행 권장(DATASET.md §5.1).

## 참고

- 스펙: `classifier/spec_5k.json` · 큐레이터: `classifier/curate.py`(--dry-run 추가)
- 이전 풀: `gs://qi-ucsd-speech-usw2/_archive/curated_v1_8k/`
- raw 재고 근거: `gs://qi-ucsd-speech-usw2/reports/globe_report.json`,
  `saa_report.json`
- 관련: `DATASET.md` §3(카운트)·§4(레시피), `src/config.py`(TARGET_PER_CLASS)
