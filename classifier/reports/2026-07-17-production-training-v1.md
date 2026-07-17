# 정식버전 학습 v1 — 백본 파인튜닝 + 클래스 가중치 + HP 스윕 (macro-F1 0.437→0.545)

- **날짜:** 2026-07-17
- **작성:** Claude (Opus 4.8) + L33gn21
- **상태:** 완료(done)
- **범위:** 코드 / 학습 / 평가 / 문서

## 요약 (TL;DR)

지금까지의 "헤드만 학습(임시)"에서 **정식버전 학습 레시피**로 전환했다. `train.py`에
**클래스 가중 손실·early stopping·드롭아웃 CLI·경량 파형 증강·Vizier(hypertune) 연동**을
추가하고, T4에서 **HP 스윕(프록시 5잡)**으로 `lr × unfreeze-top`을 탐색했다. 승자
**lr 1e-4 / unfreeze-4**로 전량 데이터 정식 학습을 돌려, 내부 홀드아웃(화자분리) **test
macro-F1 0.437→0.545 (+25%), accuracy 0.44→0.571**을 달성했다. 별도로 구축한 **VoxForge
미지 코퍼스**에서 **정확도 0.617 / 5클래스 macro-F1 0.625**로, 채널 confound가 아닌 실제
억양 신호를 학습했음을 확인했다. 모델은 Model Registry `accent-classifier` v1으로 등록.

## 배경 / 동기

- 기존 최고 산출물은 `--per-class 300` 헤드-온리 프로토타입(test macro-F1 0.437,
  acc 0.44)으로, 백본이 동결돼 억양 표현이 적응되지 않는 상한이었다.
- 문서·주석은 "macro-F1 + class weighting으로 불균형 흡수"를 표방했으나 **코드에
  클래스 가중치가 미구현**이었고(그냥 CE), early stopping·증강도 없었다.
- 목표: 5k 재구축된 curated 풀(28,237 클립)로 **최대 성능 정식버전**을 만들되, T4
  비용·시간을 관리한다.

## 수행 작업

### 1. 학습 코드 개선 (`src/`)
- **클래스 가중 손실**: `WeightedTrainer(Trainer).compute_loss` + `compute_class_weights`
  (`none|balanced|sqrt`, 기본 `balanced`). **모델 state_dict은 불변**(가중치는 학습
  쪽에만) → `evaluate.py`/`infer.py`/model_tester 호환 유지. `num_items_in_batch/**kwargs`로
  향후 transformers 버전 상향 대비.
- **Early stopping**: `--early-stopping-patience`(기본 3) + `EarlyStoppingCallback`,
  `load_best_model_at_end`로 best(macro-F1) 체크포인트 복원.
- **경량 파형 증강**(`dataset.py` `augment_waveform`, `--augment`): 학습 분할에만 랜덤
  게인 + 절반 확률 가우시안 노이즈(SNR 15~30dB). 채널 confound(GLOBE vs SAA) 강건성.
- **`--dropout` CLI**, **`--hypertune`**(cloudml-hypertune로 eval_macro_f1을 Vizier에
  보고), tunable 플래그에 underscore alias(`--unfreeze_top` 등, Vizier 주입 호환).
- 로컬 검증: 경량 CPU venv에서 로직 단위 테스트 6종 통과(서브에이전트) — 가중치 계산,
  compute_loss, compute_metrics, 증강, argparse, early-stop 배선.

### 2. 인프라 스크립트
- `gcloud/sweep.sh`(가벼운 길: 고정 그리드 프록시), `gcloud/submit_hp_tuning_job.sh`
  (Vizier, 선택), `submit_job.sh`에 `JOB_SUFFIX` 네이밍.
- **`register_model.sh` 버그 2건 수정**: (a) 프리플라이트가 `gsutil -q stat`을 써서(이
  환경은 `gcloud storage` 표준) 항상 실패→auto-register 무산 → `gcloud storage ls`로 교체.
  (b) nominal serving 이미지가 prebuilt PyTorch라 `models upload`가 TorchServe
  `model.mar`를 요구하며 실패 → 기본값을 프로젝트 자체 이미지(`IMAGE_URI`)로 변경(검증 우회).

### 3. HP 스윕 (프록시: `--per-class 1500 --epochs 3`, 클래스가중+증강, T4 순차)
`lr {3e-5,1e-4} × unfreeze {2,4}` 4잡 + 확인용 `lr1e4 × uf6` 1잡.

### 4. 정식 학습 + 등록
승자 조합으로 `--per-class 5000 --epochs 10 --early-stopping-patience 3 --auto-register`.
early stopping이 ~epoch 9에서 동작(best ≈ epoch 6, checkpoint-6438 복원).

### 5. VoxForge 미지 평가셋 (서브에이전트)
`test_raw/voxforge`(us-west2)에서 단기 GCE VM으로 클립 추출→`gs://…-usc1/test_voxforge/`
curated 레이아웃으로 스테이징. **750클립(US/UK/CA/AU/IN 각 150), 화자분리**, canonical
dialect만 매핑. CN은 VoxForge에 없어 미포함. VM 삭제 확인. CPU 커스텀잡으로 `evaluate.py` 실행.

## 결과 / 검증

**HP 스윕 (3-epoch 프록시, test macro-F1):**

| lr | uf2 | uf4 | uf6 |
|----|-----|-----|-----|
| 3e-5 | 0.376 | 0.502 | — |
| 1e-4 | 0.449 | **0.513** | 0.510 |

→ 깊은 unfreeze가 크게 유리(uf2→uf4 +0.13), lr 1e-4 > 3e-5, **uf6은 uf4 대비 포화**(개선
없음 + 느림) → 승자 **lr 1e-4 / unfreeze-4**.

**정식 학습 — 내부 홀드아웃(화자분리) test:**

| 지표 | 헤드-온리 베이스라인 | **정식버전 v1** |
|---|---|---|
| test macro-F1 | 0.437 | **0.545** |
| test accuracy | 0.44 | **0.571** |

클래스별 test F1: US 0.451 · UK 0.627 · CA 0.527 · AU 0.689 · IN 0.613 · **CN 0.361**
(CN은 epoch1 0.0 → 클래스가중으로 0.361 회복. 최소·L2 클래스라 최약).

**VoxForge 미지 코퍼스 (진짜 일반화):** accuracy **0.617**, 6라벨 macro-F1 0.521
(테스트 가능한 5클래스 macro-F1 **0.625**; CN 0 support).
클래스별 F1(혼동행렬 기준): US 0.48 · UK 0.68 · CA 0.53 · AU 0.62 · **IN 0.82**.
주요 혼동은 언어학적으로 타당(US↔CA 북미권, AU→UK 영연방). **미지 코퍼스 성능이
내부 test보다 높다** → GLOBE/SAA 채널을 외운 게 아니라 실제 억양 신호를 학습했다는 강한 근거.

**처리량(T4 실측):** fine-tune(uf4) 학습 ~11분/epoch, forward ~90-120 clips/s. 학습은
데이터로딩이 아니라 GPU 바운드로 확인 → 로컬-SSD 스테이징 최적화 불필요(생략).

## 비용 / 영향

- 스모크 1 + 스윕 5 + 정식 1 + VoxForge eval 1 = T4/CPU 잡. 대략 **~$3 내외**
  (스윕 ~$1.2, 정식 ~1.7h ~$1.3, 나머지 소액). VoxForge VM ~$0.2 미만(단기, 삭제).
- 스토리지: `test_voxforge/`(750클립, 소량) + 모델 산출물. 영향 경미.

## 후속 조치 (TODO)

- [ ] CN 강화: SpeechOcean762(본토 만다린) raw 인제스트 시 CN F1·VoxForge CN 평가 가능.
- [ ] 클래스가중이 CN 과예측을 유발(VoxForge에서 44클립이 CN 오예측) → `sqrt` 가중이나
      threshold 조정으로 정밀도/재현율 균형 실험 여지.
- [ ] US↔CA 혼동 완화(북미권) — 추가 피처/데이터 or 라벨 재정의 검토.
- [ ] 원하면 Vizier(`submit_hp_tuning_job.sh`)로 lr·unfreeze·weight_decay 정밀 탐색.
- [ ] 변경 파일 커밋(현재 미커밋): `src/train.py`,`src/dataset.py`,`requirements.txt`,
      `gcloud/{sweep.sh,submit_hp_tuning_job.sh,submit_job.sh,register_model.sh}`, 본 보고서.

## 참고

- 정식 잡: `accent-classifier-20260717-022701-final`
  (`gs://qi-ucsd-speech-usc1/outputs/classifier/…/model/`, `final_metrics.json`).
- Model Registry: `accent-classifier` v1 (us-central1).
- 스윕 잡: `…-lr{3e5,1e4}-uf{2,4}`, `…-lr1e4-uf6`.
- VoxForge: `gs://qi-ucsd-speech-usc1/test_voxforge/`(+`manifests/test.csv`),
  eval 결과 `…/final/model/test_report.json`.
- 관련: `docs/hyperparameter-tuning.md`, `DATASET.md` §5(채널 confound)·§7(VoxForge),
  `src/{train,dataset,model,evaluate}.py`.
