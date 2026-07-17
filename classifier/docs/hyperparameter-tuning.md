# 하이퍼파라미터 튜닝 가이드 (accent classifier)

> 이 분류기에서 **무엇을 조정할 수 있고**, 그걸 **GCP로 자동 최적화**하려면
> 무엇이 필요한지 정리한 문서. 코드 변경은 아직 하지 않았다 — 지금은
> "무엇을 튜닝할지 / 자동화하려면 무엇을 손봐야 할지"를 기록해 두는 단계다.
> 학습 실행 자체는 [`vertex-ai-options.md`](./vertex-ai-options.md) 참고.

---

## 0. 요약 (먼저 읽을 것)

- **튜닝 대상**은 이미 `src/train.py` argparse에 대부분 노출돼 있다 (`--lr`,
  `--unfreeze-top`, `--weight-decay`, `--warmup-ratio`, `--epochs`, `--per-class` 등).
- 우선순위: **`--lr` → `--unfreeze-top` → `--weight-decay` / `--warmup-ratio`** 순.
- `--epochs`는 `load_best_model_at_end` + `metric_for_best_model="macro_f1"` 덕에
  좀 크게 줘도 best 체크포인트가 자동 선택되므로 우선순위가 낮다.
- **GCP 자동 최적화 서비스는 있다**: **Vertex AI Hyperparameter Tuning**
  (내부적으로 **Vizier**, 베이지안 블랙박스 최적화). 지금 쓰는 Custom Job과
  **같은 컨테이너**를 그대로 쓴다.
- 다만 **코드 한 군데를 추가해야 동작한다** — trial 점수를 Vizier가 읽도록
  `cloudml-hypertune`으로 지표를 보고해야 한다 (§3). 지금은 TensorBoard와
  `final_metrics.json`에만 쓰고 있어 그대로는 HP 튜닝이 점수를 못 읽는다.

---

## 1. 튜닝 가능한 하이퍼파라미터

출처: `src/train.py:112-131` (argparse), `src/model.py:87` (dropout), `src/config.py`.

| 인자 | 기본값 | 역할 | 튜닝 가치 | 권장 탐색 범위 |
|---|---|---|---|---|
| `--lr` | `1e-4` | 학습률 | ★★★ 제일 중요 | `1e-5 ~ 3e-4` (log scale) |
| `--unfreeze-top` | `0` (헤드만) | 백본 상위 N개 트랜스포머 레이어 미세조정 | ★★★ 성능 좌우 큼 | `0, 2, 4, 6` |
| `--weight-decay` | `0.01` | L2 정규화 | ★★ | `0.0 ~ 0.1` |
| `--warmup-ratio` | `0.1` | 워밍업 비율 (특히 unfreeze 시 중요) | ★★ | `0.0 ~ 0.2` |
| `--epochs` | `8` | 에포크 수 | ★ (early-stop이 흡수) | `6 ~ 12` |
| `--batch-size` / `--grad-accum` | `8` / `2` (유효 배치 16) | 배치 크기 | ★ (주로 메모리 제약) | 고정 권장 |
| `--per-class` | `5000` | 클래스당 상한(언더샘플링) | ★ 데이터 쪽 knob | 데이터 실험용, HP 튜닝엔 고정 |
| `dropout` | `0.1` | 헤드 드롭아웃 | ★ | `0.0 ~ 0.3` |

### 함정: `dropout`은 아직 CLI로 안 빠져 있다
`dropout=0.1`은 `AccentClassifier.__init__`에만 있고 (`src/model.py:87`)
argparse 인자가 없다. 튜닝 대상에 넣으려면 `train.py`에 `--dropout` 인자를 추가하고
`AccentClassifier(MODEL_NAME, dropout=args.dropout)`로 전달하도록 고쳐야 한다.

### 참고: 왜 `--epochs`는 우선순위가 낮은가
`training_args`에 `load_best_model_at_end=True`, `metric_for_best_model="macro_f1"`,
`greater_is_better=True`가 걸려 있어 (`src/train.py:223-225`), 에포크를 넉넉히 줘도
학습 종료 시 **macro-F1이 가장 좋았던 체크포인트**를 자동으로 되살린다. 그래서
에포크는 "충분히 크게"만 주면 되고, 정밀 탐색 대상은 아니다.

---

## 2. 최적화 목표(metric)

현재 모델 선택 기준과 동일하게 **macro-F1**을 목표로 둔다.

- `compute_metrics`가 `macro_f1`(클래스별 F1의 단순 평균)을 계산한다 (`src/train.py:63-77`).
- 클래스 불균형(US/UK/CA ~6k vs CN ~1.17k)이 크므로 accuracy 대신 macro-F1이 옳다.
- HP 튜닝 시에도 이 값을 **최대화(maximize)** 목표로 준다.

---

## 3. GCP 자동 최적화 — Vertex AI Hyperparameter Tuning (Vizier)

### 무엇인가
- Custom Job과 **같은 컨테이너**로 여러 trial을 (병렬로) 실행하고, 각 trial의
  하이퍼파라미터 조합을 **Vizier**(Google의 베이지안 블랙박스 옵티마이저)가
  이전 결과를 보고 골라준다. grid/random보다 적은 trial로 좋은 조합에 수렴.
- 잡 타입만 `custom-jobs`에서 `hyperparameter-tuning-jobs`로 바뀐다.

### 동작에 필요한 **코드 변경** (필수)
지금 `train.py`는 지표를 TensorBoard와 `final_metrics.json`에만 쓴다. Vizier가
각 trial 점수를 읽으려면 `cloudml-hypertune`으로 **다시 보고**해야 한다:

```python
# train.py 안, 최종 val 지표를 얻은 뒤 (val_metrics 계산 직후)
import hypertune
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag="macro_f1",       # study spec의 metricId와 일치시킬 것
    metric_value=float(val_metrics["eval_macro_f1"]),
)
```

추가로:
- `Dockerfile`(또는 requirements)에 **`cloudml-hypertune`** 패키지 추가.
- (선택) `--dropout` 인자 노출 — §1의 함정 참고.
- Vizier는 각 trial에 하이퍼파라미터를 **CLI 인자**(예: `--lr=0.00007`)로 주입하므로,
  `train.py`가 이미 이 인자들을 받도록 돼 있는 건 그대로 호환된다.

### 제출 방식
`gcloud/submit_job.sh`(Custom Job)와 별개로, study spec을 담은
`hyperparameter-tuning-jobs` 잡을 제출한다. study spec에는 대략:

- `metricId: macro_f1`, `goal: MAXIMIZE`
- 파라미터별 타입/범위:
  - `lr`: `DOUBLE`, `1e-5 ~ 3e-4`, `scaleType: UNIT_LOG_SCALE`
  - `unfreeze_top`: `DISCRETE`, `[0, 2, 4, 6]`
  - `weight_decay`: `DOUBLE`, `0.0 ~ 0.1`
- `maxTrialCount`, `parallelTrialCount`, `algorithm`(기본 = Bayesian/Vizier)

를 넣는다. (실제 스크립트는 아직 작성 전 — 필요 시 `gcloud/submit_hp_tuning_job.sh`로 추가.)

---

## 4. 비용 주의 (CLAUDE.md §3)

HP 튜닝 비용 ≈ **trial 개수 × 1회 학습 비용**. T4 시간이 곱으로 늘어난다.
(예: 20 trial × 학습 1시간 = T4 20시간어치.)

현실적인 시작 설정:
- 탐색 파라미터를 **2~3개**로 좁힌다 (`lr`, `unfreeze-top`, `weight-decay`).
- `maxTrialCount` 12~20, `parallelTrialCount` 2~3 (병렬 = 쿼터·비용과 직결).
- 각 trial은 짧게: `--epochs`를 줄이거나 `--per-class`를 낮춰 프록시 학습 후,
  나온 최적 조합으로 **전량·풀 에포크 1회 정식 학습**.
- 제출 전 예상 시간·비용을 먼저 보고한다 (CLAUDE.md §3 규칙).
- us-west2 **T4 병렬 쿼터**를 먼저 확인한다 — `parallelTrialCount`가 쿼터를 넘으면
  trial이 대기하며 벽시계 시간만 늘어난다.

---

## 5. 두 갈래 실행 옵션

| | 가벼운 길 (수동) | 제대로 된 길 (Vizier 자동) |
|---|---|---|
| 방법 | `lr`/`unfreeze-top` 몇 조합을 병렬 Custom Job으로 수동 제출 후 비교 | Vertex HP Tuning Job으로 Vizier가 자동 탐색 |
| 코드 변경 | **없음** (지금 코드 그대로) | `hypertune` 보고 추가 + `cloudml-hypertune` 의존성 + (선택) `--dropout` + HP 잡 스크립트 |
| 비용 | 내가 정한 N개만 | trial 수만큼 (베이지안이라 효율적) |
| 언제 | 빠르게 감 잡을 때, 조합이 몇 개 안 될 때 | 파라미터 3개+를 제대로 최적화할 때 |

> 권장: 먼저 **가벼운 길**로 `lr` × `unfreeze-top` 몇 조합(예: 2×3=6개)을 돌려
> 대략 지형을 본 뒤, 가치가 확인되면 **Vizier 자동 탐색**으로 승격한다.

---

## 6. 관련 파일 맵

| 파일 | 이 문서와의 관계 |
|---|---|
| `src/train.py` | 튜닝 대상 argparse 정의 (§1). HP 튜닝 시 `hypertune` 보고 추가 지점 (§3) |
| `src/model.py` | `dropout` 기본값 위치 — 아직 CLI 미노출 (§1 함정) |
| `src/config.py` | `TARGET_PER_CLASS`(=`--per-class` 기본값) 등 데이터 knob |
| `gcloud/submit_job.sh` | 현재 Custom Job 제출 (HP 튜닝은 별도 잡 필요, §3) |
| `Dockerfile` | HP 튜닝 시 `cloudml-hypertune` 추가 지점 |
| `docs/vertex-ai-options.md` | 잡 종류 표에서 HP Tuning Job을 "나중에 선택적"으로 언급 (§③) |
