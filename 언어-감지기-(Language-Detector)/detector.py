import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. 음성 패턴(MFCC) 추출 함수
# 목소리의 고유한 지문(주파수 특성)을 뽑아내는 핵심 역할이야.
def extract_features(file_path):
    try:
        # sr=16000으로 고정해서 모든 오디오의 샘플링 레이트를 통일함
        audio, sample_rate = librosa.load(file_path, sr=16000)
        # 40개의 MFCC 특징을 추출하고, 시간축에 대해 평균을 내서 1차원 배열로 만듦
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# 2. 모델 학습 함수
def train_model(data_dir, model_save_path="voice_model.pkl"):
    print("🚀 데이터 로딩 및 패턴 추출 시작...")
    features = []
    labels = []
    
    # 클래스 정의 (0: 진짜 사람, 1: AI 합성)
    classes = {"real": 0, "fake": 1}
    
    for label_name, label_idx in classes.items():
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.exists(folder_path):
            print(f"❌ 폴더가 없어! 경로를 확인해: {folder_path}")
            return
        
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(folder_path, filename)
                data = extract_features(file_path)
                if data is not None:
                    features.append(data)
                    labels.append(label_idx)
                    
    X = np.array(features)
    y = np.array(labels)
    
    if len(X) == 0:
        print("❌ 학습할 오디오 데이터가 없습니다!")
        return

    print(f"✅ 총 {len(X)}개의 데이터 추출 완료. AI 모델 학습을 시작합니다!")
    
    # 패턴을 분류할 AI 머신러닝 모델 생성 및 학습
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    # 학습 정확도 확인 (학습 데이터에 대한 정확도)
    predictions = clf.predict(X)
    print(f"🎯 학습 데이터 정확도: {accuracy_score(y, predictions) * 100:.2f}%")
    
    # 학습된 뇌(모델)를 파일로 저장
    joblib.dump(clf, model_save_path)
    print(f"💾 모델 저장 완료: {model_save_path}")

# 3. 새로운 음성 예측 함수
def predict_voice(file_path, model_path="voice_model.pkl"):
    if not os.path.exists(model_path):
        print("❌ 학습된 모델이 없어! 먼저 학습(train)을 진행해야 해.")
        return
    
    print(f"\n🔍 분석 중: {file_path}")
    # 저장된 모델 불러오기
    clf = joblib.load(model_path)
    
    # 들어온 음성 파일의 패턴 추출
    features = extract_features(file_path)
    if features is None:
        return
    
    # 예측 수행
    features = features.reshape(1, -1)
    prediction = clf.predict(features)[0]
    probabilities = clf.predict_proba(features)[0]
    
    # 결과 출력
    classes = {0: "진짜 사람 (Real)", 1: "AI 합성 (Fake)"}
    result = classes[prediction]
    
    print("="*40)
    print(f"🚨 분석 결과: [{result}]")
    print(f"📊 확률 -> Real: {probabilities[0]*100:.1f}%, Fake: {probabilities[1]*100:.1f}%")
    print("="*40)

# 4. 실행 블록
if __name__ == "__main__":
    # 1단계: 모델 학습 (주석을 풀거나 묶어서 제어해)
    train_model(data_dir="data")
    
    # 2단계: 테스트할 파일이 있다면 예측 실행
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        predict_voice(test_file)
    else:
        print(f"💡 테스트할 '{test_file}' 파일이 없어서 예측은 건너뜁니다.")
