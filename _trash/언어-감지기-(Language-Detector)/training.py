!pip install -q numpy librosa scikit-learn joblib requests soundfile audioread
%%writefile /content/voice_detector_hf_colab.py
# -*- coding: utf-8 -*-

import os
import re
import json
import base64
import argparse
from pathlib import Path
from urllib.parse import urljoin

import requests
import numpy as np
import librosa
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HF_FIRST_ROWS_API = (
    "https://datasets-server.huggingface.co/first-rows"
    "?dataset=unfake%2Ffake_voices&config=default&split=train"
)

HF_BASE_URL = "https://datasets-server.huggingface.co"

SUPPORTED_AUDIO_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".m4a",
    ".webm",
    ".flac",
    ".ogg",
)

DEFAULT_DATA_DIR = "/content/data"
DEFAULT_MODEL_PATH = "/content/voice_model.pkl"


def safe_filename(name: str, fallback: str = "audio") -> str:
    name = str(name or fallback)
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    name = name.strip("._")
    return name or fallback


def ensure_data_dirs(data_dir: str = DEFAULT_DATA_DIR) -> None:
    Path(data_dir, "real").mkdir(parents=True, exist_ok=True)
    Path(data_dir, "fake").mkdir(parents=True, exist_ok=True)


def print_status(data_dir: str = DEFAULT_DATA_DIR) -> None:
    ensure_data_dirs(data_dir)

    data_dir_path = Path(data_dir)
    real_dir = data_dir_path / "real"
    fake_dir = data_dir_path / "fake"

    real_files = [
        p for p in real_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]

    fake_files = [
        p for p in fake_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]

    print("=" * 70)
    print("📁 데이터 폴더 상태")
    print(f"Data dir : {data_dir_path.resolve()}")
    print(f"Real dir : {real_dir.resolve()}")
    print(f"Fake dir : {fake_dir.resolve()}")
    print(f"Real audio count : {len(real_files)}")
    print(f"Fake audio count : {len(fake_files)}")
    print("=" * 70)


def call_first_rows_api(api_url: str = HF_FIRST_ROWS_API) -> dict:
    print("🌐 Hugging Face first-rows API 호출")
    print(api_url)

    response = requests.get(api_url, timeout=60)
    print(f"HTTP status: {response.status_code}")

    if response.status_code != 200:
        print("❌ API 호출 실패")
        print(response.text[:2000])
        response.raise_for_status()

    data = response.json()
    rows = data.get("rows", [])

    print(f"✅ rows count: {len(rows)}")
    return data


def find_audio_candidates(obj):
    candidates = []

    if isinstance(obj, dict):
        possible_url = (
            obj.get("src")
            or obj.get("url")
            or obj.get("download_url")
            or obj.get("path")
        )

        possible_type = (
            obj.get("type")
            or obj.get("mime_type")
            or obj.get("content_type")
        )

        if possible_url:
            possible_url_str = str(possible_url)

            is_likely_audio = (
                possible_url_str.startswith("data:audio")
                or any(ext in possible_url_str.lower().split("?")[0] for ext in SUPPORTED_AUDIO_EXTENSIONS)
                or possible_type is not None
            )

            if is_likely_audio:
                candidates.append({
                    "url": possible_url_str,
                    "mime_type": possible_type,
                    "raw": obj,
                })

        for value in obj.values():
            candidates.extend(find_audio_candidates(value))

    elif isinstance(obj, list):
        for item in obj:
            candidates.extend(find_audio_candidates(item))

    elif isinstance(obj, str):
        s = obj
        if s.startswith("data:audio") or any(ext in s.lower().split("?")[0] for ext in SUPPORTED_AUDIO_EXTENSIONS):
            candidates.append({
                "url": s,
                "mime_type": None,
                "raw": obj,
            })

    return candidates


def inspect_api(api_url: str = HF_FIRST_ROWS_API) -> None:
    data = call_first_rows_api(api_url)
    rows = data.get("rows", [])

    if not rows:
        print("⚠️ rows가 비어 있습니다.")
        return

    first_item = rows[0]
    row = first_item.get("row", first_item)

    print("\n첫 번째 row의 key 목록:")
    for key in row.keys():
        print(f"- {key}")

    print("\n첫 번째 row 미리보기:")
    print(json.dumps(row, indent=2, ensure_ascii=False)[:5000])

    candidates = find_audio_candidates(row)

    print("\n찾은 audio 후보:")
    if not candidates:
        print("- 없음")
    else:
        for i, c in enumerate(candidates[:5], start=1):
            print(f"{i}. url/src: {str(c.get('url'))[:200]}")
            print(f"   mime_type: {c.get('mime_type')}")


def normalize_url(url: str | None) -> str | None:
    if not url:
        return None

    url = str(url)

    if url.startswith("data:"):
        return url

    if url.startswith("http://") or url.startswith("https://"):
        return url

    if url.startswith("/"):
        return urljoin(HF_BASE_URL, url)

    return urljoin(HF_BASE_URL + "/", url)


def guess_extension(url: str | None = None, mime_type: str | None = None) -> str:
    if mime_type:
        mt = str(mime_type).lower()

        if "wav" in mt:
            return ".wav"
        if "mpeg" in mt or "mp3" in mt:
            return ".mp3"
        if "webm" in mt:
            return ".webm"
        if "ogg" in mt:
            return ".ogg"
        if "flac" in mt:
            return ".flac"
        if "m4a" in mt or "mp4" in mt:
            return ".m4a"

    if url:
        clean = str(url).split("?")[0]
        suffix = Path(clean).suffix.lower()
        if suffix in SUPPORTED_AUDIO_EXTENSIONS:
            return suffix

    return ".wav"


def save_data_url(data_url: str, out_path: Path) -> bool:
    try:
        _, encoded = data_url.split(",", 1)
        audio_bytes = base64.b64decode(encoded)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio_bytes)

        return out_path.exists() and out_path.stat().st_size > 0

    except Exception as e:
        print(f"❌ data URL 저장 실패: {e}")
        return False


def download_url_to_file(url: str, out_path: Path) -> bool:
    url = normalize_url(url)

    if not url:
        return False

    if url.startswith("data:audio"):
        return save_data_url(url, out_path)

    try:
        with requests.get(url, stream=True, timeout=120) as response:
            print(f"GET {url[:160]}... -> {response.status_code}")

            if response.status_code != 200:
                print(response.text[:500])
                return False

            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)

        return out_path.exists() and out_path.stat().st_size > 0

    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return False


def download_fake_samples(
    api_url: str = HF_FIRST_ROWS_API,
    out_dir: str = "/content/data/fake",
    max_rows: int = 20,
) -> None:
    ensure_data_dirs(DEFAULT_DATA_DIR)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    data = call_first_rows_api(api_url)
    rows = data.get("rows", [])

    if not rows:
        print("❌ API rows가 비어 있어서 다운로드할 수 없습니다.")
        return

    saved = 0
    skipped = 0

    print("\n⬇️ fake audio 다운로드 시작")
    print(f"저장 위치: {out_dir_path.resolve()}")
    print(f"최대 rows: {max_rows}")

    for item in rows[:max_rows]:
        row_idx = item.get("row_idx", saved)
        row = item.get("row", item)

        candidates = find_audio_candidates(row)

        if not candidates:
            print(f"⚠️ row {row_idx}: audio 후보를 찾지 못했습니다.")
            skipped += 1
            continue

        candidate = candidates[0]
        url = candidate.get("url")
        mime_type = candidate.get("mime_type")
        ext = guess_extension(url=url, mime_type=mime_type)

        filename = safe_filename(f"hf_unfake_row_{row_idx}") + ext
        out_path = out_dir_path / filename

        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"✅ 이미 존재: {out_path}")
            saved += 1
            continue

        ok = download_url_to_file(url, out_path)

        if ok:
            print(f"✅ 저장 완료: {out_path}")
            saved += 1
        else:
            print(f"❌ 저장 실패: row {row_idx}")
            skipped += 1

    print("\n" + "=" * 70)
    print("다운로드 결과")
    print(f"saved  : {saved}")
    print(f"skipped: {skipped}")
    print("=" * 70)

    print_status(DEFAULT_DATA_DIR)


def extract_features(file_path: str, sr: int = 16000):
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)

        if audio is None or len(audio) == 0:
            print(f"⚠️ 빈 오디오 파일: {file_path}")
            return None

        duration = librosa.get_duration(y=audio, sr=sample_rate)

        if duration < 1.0:
            print(f"⚠️ 너무 짧은 오디오: {file_path} ({duration:.2f}s)")
            return None

        rms = float(np.mean(librosa.feature.rms(y=audio)))

        if rms < 0.001:
            print(f"⚠️ 무음 또는 너무 작은 소리: {file_path}")
            return None

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        zcr = librosa.feature.zero_crossing_rate(audio)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)

        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [np.mean(spectral_centroid), np.std(spectral_centroid)],
            [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
            [np.mean(zcr), np.std(zcr)],
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1),
        ]).astype(np.float32)

        return features

    except Exception as e:
        print(f"❌ 특징 추출 실패: {file_path}")
        print(e)
        return None


def collect_audio_files(folder_path: str | Path):
    folder_path = Path(folder_path)

    if not folder_path.exists():
        return []

    files = []

    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS):
                files.append(Path(root) / filename)

    return sorted(files)


def load_dataset(data_dir: str = DEFAULT_DATA_DIR):
    ensure_data_dirs(data_dir)

    data_dir_path = Path(data_dir)

    classes = {
        "real": 0,
        "fake": 1,
    }

    features = []
    labels = []

    print("🚀 데이터 로딩 및 특징 추출 시작")
    print(f"데이터 경로: {data_dir_path.resolve()}")

    for class_name, label in classes.items():
        folder = data_dir_path / class_name
        audio_files = collect_audio_files(folder)

        print(f"\n📂 {class_name}: {len(audio_files)}개 파일 발견")

        for file_path in audio_files:
            feat = extract_features(str(file_path))

            if feat is not None:
                features.append(feat)
                labels.append(label)

    if not features:
        return None, None

    return np.array(features), np.array(labels)


def train_model(
    data_dir: str = DEFAULT_DATA_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
) -> None:
    X, y = load_dataset(data_dir)

    if X is None or y is None:
        print("❌ 학습할 데이터가 없습니다.")
        print("먼저 /content/data/real 과 /content/data/fake 폴더에 오디오 파일을 넣어주세요.")
        return

    real_count = int(np.sum(y == 0))
    fake_count = int(np.sum(y == 1))

    print("\n" + "=" * 70)
    print("데이터 요약")
    print(f"전체 샘플 수 : {len(y)}")
    print(f"Real 샘플 수 : {real_count}")
    print(f"Fake 샘플 수 : {fake_count}")
    print("=" * 70)

    if real_count == 0 or fake_count == 0:
        print("❌ real/fake 두 클래스가 모두 있어야 학습할 수 있습니다.")
        print("현재 Hugging Face unfake/fake_voices API는 fake 샘플만 제공합니다.")
        print("따라서 /content/data/real/ 폴더에 실제 사람 음성을 업로드해야 합니다.")
        return

    min_class_count = min(real_count, fake_count)

    if len(y) >= 10 and min_class_count >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    else:
        print("⚠️ 데이터가 너무 적어서 전체 데이터로 학습/평가합니다.")
        X_train, X_test, y_train, y_test = X, X, y, y

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    print("\n🧠 모델 학습 시작...")
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print("\n🎯 학습 정확도")
    print(f"{accuracy_score(y_train, train_pred) * 100:.2f}%")

    print("\n🧪 테스트 정확도")
    print(f"{accuracy_score(y_test, test_pred) * 100:.2f}%")

    print("\n📊 Classification Report")
    print(classification_report(
        y_test,
        test_pred,
        target_names=["Real", "Fake"],
        zero_division=0,
    ))

    print("\n📌 Confusion Matrix")
    print(confusion_matrix(y_test, test_pred))

    save_obj = {
        "model": model,
        "labels": {
            0: "진짜 사람 음성 Real",
            1: "AI 합성 음성 Fake",
        },
        "feature_type": "mfcc_spectral_chroma",
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(save_obj, model_path)

    print(f"\n💾 모델 저장 완료: {model_path}")


def predict_voice(
    file_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
) -> None:
    file_path = Path(file_path)
    model_path = Path(model_path)

    if not model_path.exists():
        print("❌ 모델 파일이 없습니다.")
        print("먼저 학습을 실행하세요.")
        return

    if not file_path.exists():
        print(f"❌ 분석할 파일이 없습니다: {file_path}")
        return

    saved = joblib.load(model_path)
    model = saved["model"]
    labels = saved["labels"]

    feat = extract_features(str(file_path))

    if feat is None:
        print("❌ 특징 추출 실패로 예측할 수 없습니다.")
        return

    feat = feat.reshape(1, -1)

    pred = model.predict(feat)[0]
    probs = model.predict_proba(feat)[0]

    real_prob = float(probs[0] * 100)
    fake_prob = float(probs[1] * 100)
    confidence = max(real_prob, fake_prob)

    if confidence >= 80:
        confidence_label = "High"
    elif confidence >= 60:
        confidence_label = "Medium"
    else:
        confidence_label = "Low / Uncertain"

    print("=" * 70)
    print(f"🔍 분석 파일: {file_path}")
    print("=" * 70)
    print(f"🚨 결과: {labels[pred]}")
    print(f"Real 확률: {real_prob:.1f}%")
    print(f"Fake 확률: {fake_prob:.1f}%")
    print(f"Confidence: {confidence_label}")

    if confidence < 60:
        print("⚠️ 신뢰도가 낮습니다. 더 길고 깨끗한 음성을 사용해 주세요.")

    print("-" * 70)
    print("주의: 이 결과는 확률적 추정이며 확정적 증거가 아닙니다.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Colab baseline: Human voice vs AI-generated voice detector"
    )

    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect-api")
    inspect_parser.add_argument("--api-url", default=HF_FIRST_ROWS_API)

    download_parser = subparsers.add_parser("download-fake")
    download_parser.add_argument("--api-url", default=HF_FIRST_ROWS_API)
    download_parser.add_argument("--out", default="/content/data/fake")
    download_parser.add_argument("--max-rows", type=int, default=20)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--data", default=DEFAULT_DATA_DIR)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", default=DEFAULT_DATA_DIR)
    train_parser.add_argument("--model", default=DEFAULT_MODEL_PATH)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("file")
    predict_parser.add_argument("--model", default=DEFAULT_MODEL_PATH)

    args = parser.parse_args()

    if args.command == "inspect-api":
        inspect_api(args.api_url)

    elif args.command == "download-fake":
        download_fake_samples(
            api_url=args.api_url,
            out_dir=args.out,
            max_rows=args.max_rows,
        )

    elif args.command == "status":
        print_status(args.data)

    elif args.command == "train":
        train_model(
            data_dir=args.data,
            model_path=args.model,
        )

    elif args.command == "predict":
        predict_voice(
            file_path=args.file,
            model_path=args.model,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
  !mkdir -p /content/data/real
!mkdir -p /content/data/fake
!python /content/voice_detector_hf_colab.py status
!python /content/voice_detector_hf_colab.py inspect-api
!python /content/voice_detector_hf_colab.py download-fake --max-rows 20
from google.colab import files
import shutil
from pathlib import Path

uploaded = files.upload()

real_dir = Path("/content/data/real")
real_dir.mkdir(parents=True, exist_ok=True)

for filename in uploaded.keys():
    shutil.move(filename, real_dir / filename)

print("업로드 완료")
!python /content/voice_detector_hf_colab.py status
!python /content/voice_detector_hf_colab.py train --data /content/data
from google.colab import files

uploaded = files.upload()
test_file = list(uploaded.keys())[0]

print("테스트 파일:", test_file)
!python /content/voice_detector_hf_colab.py predict "/content/{test_file}"
