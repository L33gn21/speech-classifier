# speech-classifier

영어 발음(accent) 분류기 프로젝트. 사용자의 발화를 입력받아 미국/영국/인도 등 억양별로
근접도를 퍼센티지로 출력하는 것이 1차 목표 (예: US 30%, England 45%, Indian 10%...).

## 목표 및 로드맵

### 레벨 1 (1차 목표, 현재 진행 대상)
사전학습 오디오 인코더 + 분류 헤드 파인튜닝.
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

## 하드웨어 제약

- GPU: RTX 4060 (8GB VRAM 가정).
- 배치 사이즈 4~8 + mixed precision(fp16) + 필요시 gradient accumulation / gradient
  checkpointing으로 운용. HuggingFace Trainer 사용 시 옵션으로 처리 가능.

## 데이터셋

**Common Voice 26.0 - English (en)**, CC0-1.0, 약 88.14GB.
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
1. `validated.tsv` 로드
2. `accents`가 대상 4개 억양인 행만 필터
3. 품질 필터: `up_votes >= 2` and `down_votes == 0`
4. 억양별로 목표 개수만큼 랜덤 샘플링 (균형 맞춤)
5. **`client_id` 기준으로 train/test 분리** — 같은 화자가 train/test에 동시에 들어가면
   모델이 억양이 아니라 화자 목소리를 외워 정확도가 뻥튀기됨. 반드시 화자 단위로 split.
6. 선택된 mp3만 16kHz 모노로 리샘플, 길이 정규화(예: 5~10초로 자르기/패딩)
7. 기본 제공 train/dev/test split은 억양 균형을 고려하지 않았으므로 직접 만든 split을 사용.

