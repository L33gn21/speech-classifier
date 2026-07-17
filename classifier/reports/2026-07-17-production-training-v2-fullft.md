# 정식버전 학습 v2 — 전체 파인튜닝(full fine-tune)으로 개선 (test macro-F1 0.545→0.608)

- **날짜:** 2026-07-17
- **작성:** Claude (Opus 4.8) + L33gn21
- **상태:** 완료(done)
- **범위:** 학습 / 평가 / 하이퍼파라미터 / 코드(버그수정) / 문서

## 요약 (TL;DR)

v1(부분 unfreeze-4) 결과를 근거로 개선 실험을 돌렸다. 프록시 스윕으로 **전체 파인튜닝
(unfreeze-12) + lr 3e-5**가 uf4 정체 구간을 확실히 넘는 것을 확인하고, 전량 데이터로
정식 학습했다. 내부 홀드아웃(화자분리) **test macro-F1 0.545→0.608 (+11.6%), acc
0.571→0.620**. 미지 코퍼스 **VoxForge 정확도 0.617→0.704, 5클래스 macro-F1 0.625→0.706**.
전 클래스가 개선됐고 특히 **CN이 0.36→0.49(내부)**로 크게 올랐으며, v1에서 문제였던
**VoxForge CN 과예측(44→6클립)이 사실상 해소**됐다. 모델은 Model Registry
`accent-classifier` **v2**로 등록. 과정에서 `register_model.sh` 버그 3건도 수정.

## 배경 / 동기

- v1(uf4/lr1e-4/balanced): test macro-F1 0.545, VoxForge acc 0.617. 최약은 CN(0.36),
  그리고 VoxForge에서 balanced 가중치가 **CN 과예측**(44클립 오예측)을 유발.
- v1 스윕은 3-epoch 프록시에서 깊이가 uf4에 정체(uf4≈uf6)했으나, **전체 파인튜닝(uf12)과
  정규화/스케줄 knob은 미탐색**이었다. "최대 성능"의 남은 레버.

## 수행 작업

### 1. v2 프록시 스윕 (per-class 1500, 4 epochs, augment, T4 순차)
| config | eval_mf1 | test_mf1 | CN f1 |
|---|---|---|---|
| uf4 / lr1e-4 / **sqrt** | 0.487 | 0.507 | 0.41 |
| **uf12 / lr3e-5 / balanced** | **0.514** | 0.568 | **0.54** |
| uf12 / lr3e-5 / sqrt | 0.504 | 0.572 | 0.49 |

→ **전체 파인튜닝(uf12)이 4-epoch 프록시(깊은 구조에 불리한 조건)에서도 uf4를 확실히 상회.**
CN이 uf4 ~0.41 → uf12 ~0.5+로 급등 — 가중치 조정보다 **용량(전체 FT)이 소수 클래스를 더
잘 잡는다.** balanced≈sqrt(노이즈 범위)이나 선택 지표(eval_macro_f1)·CN에서 balanced 우위.

### 2. HP 조정 (사용자 결정: 추가 자동탐색 없이 "바로 정식 학습")
승자 구조(uf12/lr3e-5/balanced)에 **저위험 스케줄 개선 1건만 반영**: `warmup-ratio 0.1→0.15`
(깊은 FT 초기 안정화). 검증된 정규화(dropout 0.1, weight-decay 0.01)는 유지 — 미검증
변경으로 회귀 위험을 만들지 않기 위함. (Vizier 자동탐색은 인프라 준비완료·이번엔 미실행.)

### 3. 정식 학습
`--unfreeze-top=12 --lr=3e-5 --class-weight=balanced --warmup-ratio=0.15 --per-class=5000
--epochs=10 --early-stopping-patience=3 --augment --auto-register`. trainable **85.06M/94.38M**
(CNN 특징추출기만 동결). grad clip(max_grad_norm=1.0)로 초기 큰 grad_norm도 안정.
early stopping으로 best(macro-F1) 체크포인트 복원.

### 4. `register_model.sh` 버그 3건 수정 (auto-register가 전혀 동작 안 하던 원인)
- (a) 프리플라이트 `gsutil -q stat` → `gcloud storage ls` (이 환경 표준).
- (b) nominal serving 이미지가 prebuilt PyTorch라 `models upload`가 `model.mar` 요구·실패
  → 기본값을 프로젝트 자체 이미지(`IMAGE_URI`)로 (검증 우회).
- (c) 버전 추가 분기에서 `--parent-model`에 숫자 id만 전달→"Location ID not provided"
  → 전체 리소스 경로(`projects/…/locations/…/models/ID`)로 재구성.
  (v1은 첫 등록이라 이 분기를 안 타서 잠복했던 버그.)

## 결과 / 검증

**내부 홀드아웃(화자분리) test — v1 대비:**

| 지표 | v1 (uf4) | **v2 (full FT)** |
|---|---|---|
| test macro-F1 | 0.545 | **0.608** |
| test accuracy | 0.571 | **0.620** |
| US / UK / CA / AU / IN / CN (F1) | .45/.63/.53/.69/.61/.36 | **.50/.68/.57/.74/.67/.49** |

CN은 precision 0.77 / recall 0.36 — **오예측이 적고(정밀)**, 놓치는 게 많은 쪽(재현 낮음).

**VoxForge 미지 코퍼스(진짜 일반화, 750클립·5클래스) — v1 대비:**

| 지표 | v1 | **v2** |
|---|---|---|
| accuracy | 0.617 | **0.704** |
| 5클래스 macro-F1 | 0.625 | **0.706** |
| CN 오예측 클립수(true CN=0) | 44 | **6** |

클래스별 F1: US .48→.59 · UK .68→**.81** · CA .53→.57 · AU .62→.72 · IN .82→**.84**.
**미지 코퍼스에서 전 클래스 개선 + CN 과예측 해소** → 채널 confound가 아닌 실제 억양 신호
학습을 재확인. (내부 test보다 VoxForge가 높은 경향도 유지.)

**처리량/시간:** 전체 FT(uf12)는 학습 ~1.5h(전량, early stop). v1(uf4)보다 스텝당 느리나
early stopping으로 총시간 통제.

## 비용 / 영향

- v2 프록시 3 + 정식 1 + VoxForge eval 1(CPU) + 등록 재시도. 대략 **~$3 내외**.
- Model Registry: `accent-classifier` v2 추가(v1 보존). 산출물
  `gs://qi-ucsd-speech-usc1/outputs/classifier/accent-classifier-20260717-082914-v2final/`.

## 후속 조치 (TODO)

- [ ] CN **recall** 개선(현재 정밀↑재현↓): SpeechOcean762 본토 만다린 인제스트로 CN 데이터
      확충이 근본책. 또는 CN 임계값/`sqrt` 가중 재실험.
- [ ] US↔CA 혼동(북미권)은 여전한 최대 혼동원 — 추가 데이터/피처 검토.
- [ ] 준비된 Vizier(`submit_hp_tuning_job.sh`)로 weight_decay·warmup·dropout·lr 정밀
      자동탐색 시 소폭 추가 개선 여지(이번엔 시간상 생략).
- [ ] 변경 파일 커밋(미커밋): `gcloud/register_model.sh`(버그3건), 본 보고서.
      (v1 세션의 `src/*`,`requirements.txt`,`gcloud/*` 변경도 함께 미커밋 상태.)

## 참고

- 정식 잡: `accent-classifier-20260717-082914-v2final`. Registry: `accent-classifier` v2.
- 프록시: `…-v2-uf4-sqrt`, `…-v2-uf12-bal`, `…-v2-uf12-sqrt`.
- VoxForge eval: `…/v2final/model/test_report.json`(acc 0.704), 셋 구축은 v1 보고서 참고.
- 직전: [정식버전 학습 v1](./2026-07-17-production-training-v1.md).
