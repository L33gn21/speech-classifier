# speech-classifier — Level 1

wav2vec2-base backbone + linear head로 억양(us/england/indian/australia)을
분류하고 클래스별 softmax 확률을 "억양 근접도 %"로 출력한다.

## 구조
```
src/
  config.py        # 경로·라벨·하이퍼파라미터 중앙 설정
  prepare_data.py  # validated.tsv -> 균형·화자분리 매니페스트 (torch 불필요)
  dataset.py       # mp3 로드(16k mono) + Wav2Vec2 collator
  model.py         # wav2vec2 + frame-level head (Level 2 대비 frame_logits 보존)
  train.py         # HF Trainer 파인튜닝 (fp16, backbone freeze)
  infer.py         # 오디오 -> 억양 % (+ --frames frame-level 히트맵)
```

## 환경 (중요)
단일 **`.venv`** (Python 3.13 + torch 2.9.1+cu126)로 학습·추론·데이터 준비를 모두 한다.
Python 3.14는 torch가 cu130 빌드밖에 없어 이 머신 드라이버(CUDA 12.7)에서 GPU를
못 잡으므로 3.13 + cu126을 쓴다.

```bash
# venv 재현 (이미 구성됨) — uv 바이너리는 ~/.local/bin/uv
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

## 실행
```bash
# 1. 매니페스트 생성 (완료됨: data/manifests/{train,test}.csv, 16k 클립)
.venv/bin/python src/prepare_data.py            # --per-class N 로 규모 조절
#   --check-exists 로 누락 mp3 필터 가능 (현재는 전체 추출 완료)

# 2. 헤드만 학습 (backbone freeze)
.venv/bin/python src/train.py --epochs 8 --batch-size 8 --grad-accum 2

# 3. 잘 되면 상단 레이어까지 unfreeze
.venv/bin/python src/train.py --unfreeze-top 4 --lr 2e-5 --epochs 6 \
    --gradient-checkpointing

# 4. 추론
.venv/bin/python src/infer.py some_clip.mp3
.venv/bin/python src/infer.py some_clip.mp3 --plot heatmap.png   # Level 2 미리보기
```

## 설계 메모
- **화자 단위 split**: `client_id` 기준으로 train/test 분리 → 목소리 암기 방지.
- **frame-level 보존**: head를 프레임마다 적용해 `frame_logits [B,T,C]`를 만들고
  시간축 masked-mean으로 utterance logits를 얻는다. 단일 linear라 "표현 pooling 후
  head"와 수학적으로 동일하면서 Level 2(구간별 억양 히트맵) 출력을 공짜로 확보한다.
- 억양 라벨은 `accents` 서술형 문자열을 정확 매칭(혼합 `a|b` 행 제외)해 매핑한다.
