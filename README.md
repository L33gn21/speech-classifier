# speech-classifier

음성 기반 범죄 수사 보조 도구. 발화를 2단계로 처리한다.

1. **detector** — AI 합성 음성 vs 실제 사람 음성 판별 (real/fake).
2. **classifier** — 사람 음성이면 영어 억양(us/england/indian/australia)별 근접도를 %로 출력.

`발화 → [detector] AI면 종료 / 사람이면 → [classifier] 억양(국적 단서) 추정`

## 구조
```
src/
  detector/          # 1단계: 합성 음성 탐지
    model.py           # resnet18(1ch) on log-mel → real/fake
    dataset.py         # real/ + fake/ wav → 정규화 log-mel
    train.py           # 학습 → outputs/detector/detector.pt
    inference.py       # 오디오 -> "Real"/"Fake"
  classifier/        # 2단계: 억양 분류 (wav2vec2)
    config.py          # 경로·라벨·하이퍼파라미터 중앙 설정
    prepare_data.py    # validated.tsv -> 균형·화자분리 매니페스트 (torch 불필요)
    dataset.py         # mp3 로드(16k mono) + Wav2Vec2 collator
    model.py           # wav2vec2 + frame-level head (Level 2 대비 frame_logits 보존)
    train.py           # HF Trainer 파인튜닝 (fp16, backbone freeze)
    infer.py           # 오디오 -> 억양 % (+ --frames frame-level 히트맵)
    webui.py           # gradio 데모 (녹음/업로드 -> % + 히트맵)

data/
  detector/          # real/ , fake/  (합성 탐지 학습 데이터)
  classifier/en/     # Common Voice English (clips/, validated.tsv, manifests/)

outputs/
  detector/            # detector.pt
  classifier/          # 억양 분류기 (unfreeze 버전)
  classifier_headonly/ # 억양 분류기 (head-only 베이스라인)
```

경로는 모두 스크립트 파일 위치(`src/<model>/…`) 기준으로 프로젝트 루트를 잡으므로
cwd에 무관하다. 실행은 루트 디렉터리에서 한다.

## 환경 (중요)
단일 **`.venv`** (Python 3.13 + torch 2.9.1+cu126)로 두 모델의 학습·추론·데이터 준비를
모두 한다. Python 3.14는 torch가 cu130 빌드밖에 없어 이 머신 드라이버(CUDA 12.7)에서
GPU를 못 잡으므로 3.13 + cu126을 쓴다.

```bash
# venv 재현 (이미 구성됨) — uv 바이너리는 ~/.local/bin/uv
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

## 실행 — 1단계: detector
```bash
# data/detector/{real,fake}/ 를 채운 뒤 학습
.venv/bin/python src/detector/train.py          # -> outputs/detector/detector.pt

# 단일 오디오 판별
.venv/bin/python src/detector/inference.py path/to/clip.wav
```
> 주의: 현재 `data/detector`에는 fake(generated_audio)만 있고 `real/`이 비어 있다.
> 학습 전 real/fake 폴더를 채워야 한다. 자세한 내용은 `data/detector/README.md` 참고.

## 실행 — 2단계: classifier
```bash
# 1. 매니페스트 생성 -> data/classifier/manifests/{train,test}.csv
.venv/bin/python src/classifier/prepare_data.py       # --per-class N 로 규모 조절
#   --check-exists 로 누락 mp3 필터 가능

# 2. 헤드만 학습 (backbone freeze)
.venv/bin/python src/classifier/train.py --epochs 8 --batch-size 8 --grad-accum 2

# 3. 잘 되면 상단 레이어까지 unfreeze
.venv/bin/python src/classifier/train.py --unfreeze-top 4 --lr 2e-5 --epochs 6 \
    --gradient-checkpointing

# 4. 추론 (기본 --model-dir = outputs/classifier)
.venv/bin/python src/classifier/infer.py some_clip.mp3
.venv/bin/python src/classifier/infer.py some_clip.mp3 --plot heatmap.png   # Level 2 미리보기

# 5. 웹 데모
.venv/bin/python src/classifier/webui.py
```

## 설계 메모
- **탐지기 데이터 균형**: real/fake의 화자·문장 분포가 겹치게 맞춰 "합성 아티팩트"를
  학습하게 한다 (데이터셋 차이를 학습하면 안 됨). 여러 보코더를 섞을 것.
- **분류기 화자 단위 split**: `client_id` 기준으로 train/test 분리 → 목소리 암기 방지.
- **분류기 frame-level 보존**: head를 프레임마다 적용해 `frame_logits [B,T,C]`를 만들고
  시간축 masked-mean으로 utterance logits를 얻는다. 단일 linear라 "표현 pooling 후
  head"와 수학적으로 동일하면서 Level 2(구간별 억양 히트맵) 출력을 공짜로 확보한다.
- 억양 라벨은 `accents` 서술형 문자열을 정확 매칭(혼합 `a|b` 행 제외)해 매핑한다.
