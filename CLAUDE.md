# speech-classifier

음성 기반 범죄 수사 보조 도구. 입력 발화에 대해 **2단계 파이프라인**을 돌린다.

1. **탐지기(detector)** — 발화가 AI 합성 음성인지 실제 사람 음성인지 판별 (real vs fake).
2. **분류기(classifier)** — 사람 음성으로 판정된 경우에만, 영어 억양(미국/영국/인도/호주)별
   근접도를 퍼센트로 출력 (예: US 30%, England 45%, Indian 10%...).

즉 최종 흐름은 `발화 → [detector] AI면 여기서 종료 / 사람이면 → [classifier] 국적(억양) 추정`.
범죄 수사에서 "이 녹취가 딥페이크/합성 음성인가"를 먼저 거르고, 진짜 사람이면 화자의
억양권(국적 단서)을 좁히는 용도.

## 저장소 구조

두 모델은 코드·데이터·산출물이 모두 분리되어 있다.

```
src/
  detector/        # 1단계: AI 합성 음성 탐지 (WaveFake 등)
    model.py         # resnet18(1ch) on log-mel spectrogram → real/fake 2-class
    dataset.py       # real/ + fake/ wav 로드 → 128x128 정규화 log-mel
    train.py         # 학습 루프 → outputs/detector/detector.pt
    inference.py     # 단일 오디오 → "Real" / "Fake"
  classifier/      # 2단계: 억양(국적) 분류 — 상세는 아래 "분류기" 섹션
    config.py prepare_data.py dataset.py model.py train.py infer.py webui.py

data/
  detector/        # 탐지기 학습 데이터 (real/ , fake/) — 아래 "탐지기 데이터" 참고
  classifier/en/   # Common Voice English (clips/, validated.tsv, manifests/)

outputs/
  detector/            # detector.pt
  classifier/          # 억양 분류기 (backbone 상단 unfreeze 버전)
  classifier_headonly/ # 억양 분류기 (head-only 베이스라인)
```

- 경로는 각 스크립트에서 `Path(__file__)` 기준으로 프로젝트 루트를 잡는다
  (`src/<model>/*.py` → `parent.parent.parent` = 루트). cwd에 의존하지 않는다.
- 실행은 루트에서 `.venv/bin/python src/detector/train.py`,
  `.venv/bin/python src/classifier/train.py` 형태로 한다 (스크립트 디렉터리가
  자동으로 import path에 들어가므로 `from model import ...` 형태 플랫 import가 동작).

---

# 1단계: 탐지기 (detector)

AI 합성(TTS/보코더) 음성과 실제 사람 음성을 구분하는 2-class 분류기.

- 입력: 16kHz 오디오 → log-mel spectrogram (n_mels=128, 128프레임으로 crop/pad, 표준화).
- 모델: `torchvision` **resnet18** (입력 conv를 1채널로 교체, fc를 2-class로 교체).
- 라벨: `real=0`, `fake=1`. 출력 argmax로 "Real"/"Fake" 판정.
- 산출물: `outputs/detector/detector.pt`.

## 탐지기 데이터

`data/detector/` 아래에 **`real/`** 과 **`fake/`** 두 하위 폴더를 두고 각각 wav를 넣는다
(`WaveFakeDataset`이 이 두 폴더를 스캔).

- **fake**: WaveFake류 보코더 합성 음성. 현재 `data/detector/generated_audio/` 아래에
  ljspeech/jsut 기반 여러 보코더(melgan, hifiGAN, parallel_wavegan, waveglow 등) 출력이
  들어와 있다. 이것들을 `data/detector/fake/`로 모으거나 링크한다.
- **real**: 위 합성의 원본에 해당하는 진짜 녹음(LJSpeech / JSUT 원본 등)을
  `data/detector/real/`에 넣는다. fake와 화자·문장 분포가 겹치도록 맞춰야
  모델이 "합성 아티팩트"를 학습하지 "데이터셋 차이"를 학습하지 않는다.

## 탐지기 TODO / 주의

- 현재 `data/detector`에는 fake(generated_audio)만 있고 `real/`이 비어 있다.
  학습 전에 real/fake 폴더를 채워야 한다.
- `train.py`는 아직 train/test split·검증 지표가 없다 (`train_test_split` import만 있고
  전체를 학습에 사용). 실사용 전 화자/원본 단위 hold-out 평가를 추가할 것.
- 합성 탐지는 보코더 종류에 과적합되기 쉽다 → 여러 보코더를 섞고, 미학습 보코더로
  일반화 성능을 따로 확인하는 것이 중요.

---

# 2단계: 억양(국적) 분류기 (classifier)

사람 음성으로 판정된 발화의 영어 억양권을 추정한다. 사전학습 오디오 인코더 +
분류 헤드 파인튜닝.

## 목표 및 로드맵

### 레벨 1 (현재 진행 대상)
- Backbone: `facebook/wav2vec2-base` (HuggingFace). base 사이즈로 시작 (large는 8GB VRAM에 부담).
- 구조: wav2vec2 backbone → (frame-level 표현 유지) → mean pooling → linear 분류 헤드 → softmax.
- 학습 순서: backbone freeze 후 헤드만 학습 → 잘 되면 backbone 상단 레이어까지 unfreeze.
- 출력: 클래스별 softmax 확률을 그대로 "억양 근접도 퍼센트"로 사용.

### 레벨 2 (코칭 기능 확장, 비교적 저비용)
mean pooling 이전의 frame-level 분류 출력을 살려서, 발화 중 "어느 구간이 어떤 억양에
가까웠는지" 시간축 히트맵으로 시각화. **레벨 1 코드를 짤 때부터 frame-level 출력을
버리지 않는 구조로 설계할 것** (mean pooling 결과만 저장하지 말 것).

### 레벨 3 (본격 발음 코칭, 별도 확장 과제)
음소(phoneme) 단위 정렬(MFA 또는 wav2vec2 기반 phoneme 모델) + 목표 억양 음향 통계와
비교해 "이 음소를 이렇게 교정하라" 수준의 피드백. GOP(Goodness of Pronunciation) 계열
기법과 관련. 범위가 크므로 레벨 1/2 이후 별도 과제로 취급.
- 이 단계를 염두에 둔다면 "사용자가 정해진 문장을 읽는다(read speech)" 가정을 미리
  깔아두는 것을 권장 (자유 발화는 음소 정렬 난이도가 급상승).
- "표준 발음(reference)"의 정의를 미리 합의해둘 것 (예: 특정 데이터셋의 해당 억양
  화자 평균으로 잠정 정의).

## 데이터셋

**Common Voice 26.0 - English (en)**, CC0-1.0, 약 88.14GB. `data/classifier/en/` 에 위치.
(다른 언어팩, 예: Hindi 등은 억양 라벨이 아니라 별개 언어 데이터이므로 사용하지 않음.
 인도 억양 영어는 English 데이터셋 안에 accent=indian 화자로 포함되어 있음.)

- 라벨: `accents` 컬럼 (자율 입력이라 상당수 공란 — 정상).
- 확인된 억양별 클립 수 (validated 기준):
  - us: 573,393
  - england: 204,106
  - indian: 152,723
  - canada: 101,061
  - australia: 69,363
  - scotland: 68,131
  - african: 60,292
  - 그 외(뉴질랜드, 아일랜드 등): 각 1~2만 이하

### 1차 클래스 구성 (결정됨)
`us`, `england`, `indian`, `australia` 4개 억양으로 시작. (canada는 us와 음향적으로
너무 비슷해 초반 제외.) 억양당 3,000~5,000 클립으로 균형 샘플링해서 소규모로 시작
(예: 4개 × 4,000 = 16,000 클립) — 8배 가까운 불균형(us vs scotland)을 이 단계에서
under-sampling으로 해소.

### 데이터 준비 레시피
1. `data/classifier/en/validated.tsv` 로드
2. `accents`가 대상 4개 억양인 행만 필터
3. 품질 필터: `up_votes >= 2` and `down_votes == 0`
4. 억양별로 목표 개수만큼 랜덤 샘플링 (균형 맞춤)
5. **`client_id` 기준으로 train/test 분리** — 같은 화자가 train/test에 동시에 들어가면
   모델이 억양이 아니라 화자 목소리를 외워 정확도가 뻥튀기됨. 반드시 화자 단위로 split.
6. 선택된 mp3만 16kHz 모노로 리샘플, 길이 정규화(예: 5~10초로 자르기/패딩)
7. 기본 제공 train/dev/test split은 억양 균형을 고려하지 않았으므로 직접 만든 split을 사용.
   결과 매니페스트는 `data/classifier/manifests/{train,test}.csv`.

---

## 하드웨어 제약 (두 모델 공통)

- GPU: RTX 4060 (8GB VRAM 가정).
- 배치 사이즈 4~8 + mixed precision(fp16) + 필요시 gradient accumulation / gradient
  checkpointing으로 운용. HuggingFace Trainer 사용 시 옵션으로 처리 가능.
