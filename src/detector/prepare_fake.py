"""data/detector/fake/ 를 학습용 보코더 6종에서 라운드로빈으로 채운다.

- LJSpeech 발화 ID 를 6종 보코더에 균등 배정(ID 중복 없음) -> fake ~= real(13,100) 균형.
- hifiGAN 은 미학습 hold-out 으로 예약(여기서 제외; train.py 가 일반화 검증에만 사용).
- 링크 이름 '<vocoder>__<원본파일명>' 으로 출처 추적 + 충돌 방지.
- fake/ 는 이 스크립트로 언제든 재생성 가능(seed 고정).

실행: .venv/bin/python src/detector/prepare_fake.py
"""
import os
import re
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GEN = PROJECT_ROOT / "data" / "detector" / "generated_audio"
FAKE = PROJECT_ROOT / "data" / "detector" / "fake"

TRAIN_VOCODERS = [
    "ljspeech_full_band_melgan",
    "ljspeech_melgan",
    "ljspeech_melgan_large",
    "ljspeech_multi_band_melgan",
    "ljspeech_parallel_wavegan",
    "ljspeech_waveglow",
]
HOLDOUT = "ljspeech_hifiGAN"  # fake/ 에 넣지 않음 (일반화 검증용 예약)

SEED = 42
ID_RE = re.compile(r"^(LJ\d{3}-\d{4})")


def index_vocoder(voc):
    """폴더 안 wav 를 {발화ID: 파일명} 으로 인덱싱."""
    m = {}
    for f in os.listdir(GEN / voc):
        if not f.endswith(".wav"):
            continue
        mo = ID_RE.match(f)
        if mo:
            m[mo.group(1)] = f
    return m


def main():
    random.seed(SEED)

    indexes = {v: index_vocoder(v) for v in TRAIN_VOCODERS}
    # 모든 학습 보코더에 공통으로 존재하는 발화 ID (교집합)
    common = set.intersection(*(set(ix.keys()) for ix in indexes.values()))
    ids = sorted(common)
    random.shuffle(ids)
    print(f"공통 발화 ID: {len(ids)}개")

    FAKE.mkdir(parents=True, exist_ok=True)
    # 기존 링크 정리(재실행 대비)
    for f in os.listdir(FAKE):
        p = FAKE / f
        if p.is_symlink() or p.is_file():
            p.unlink()

    per_voc = {v: 0 for v in TRAIN_VOCODERS}
    for i, uid in enumerate(ids):
        voc = TRAIN_VOCODERS[i % len(TRAIN_VOCODERS)]
        src_name = indexes[voc][uid]
        rel = os.path.join("..", "generated_audio", voc, src_name)  # fake/ 기준 상대경로
        os.symlink(rel, FAKE / f"{voc}__{src_name}")
        per_voc[voc] += 1

    print(f"\nfake/ 생성 링크: {sum(per_voc.values())}개")
    for v in TRAIN_VOCODERS:
        print(f"  {v}: {per_voc[v]}")
    print(f"\nhold-out(예약, fake 미포함): {HOLDOUT}")


if __name__ == "__main__":
    main()
