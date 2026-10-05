import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Voice-pattern (MFCC) extraction function
# The core step that extracts the unique fingerprint (frequency characteristics) of a voice.
def extract_features(file_path):
    try:
        # sr=16000 is fixed so every audio file has the same sampling rate
        audio, sample_rate = librosa.load(file_path, sr=16000)
        # Extract 40 MFCC features and average them over time into a 1-D array
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# 2. Model training function
def train_model(data_dir, model_save_path="voice_model.pkl"):
    print("🚀 Loading data and extracting patterns...")
    features = []
    labels = []
    
    # Class definition (0: real human, 1: AI-synthesized)
    classes = {"real": 0, "fake": 1}
    
    for label_name, label_idx in classes.items():
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found! Check the path: {folder_path}")
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
        print("❌ No audio data to train on!")
        return

    print(f"✅ Extracted {len(X)} samples in total. Starting AI model training!")
    
    # Create and train the machine-learning model that classifies the patterns
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    # Check the training accuracy (accuracy on the training data)
    predictions = clf.predict(X)
    print(f"🎯 Training-data accuracy: {accuracy_score(y, predictions) * 100:.2f}%")
    
    # Save the trained model to a file
    joblib.dump(clf, model_save_path)
    print(f"💾 Model saved: {model_save_path}")

# 3. Prediction function for new audio
def predict_voice(file_path, model_path="voice_model.pkl"):
    if not os.path.exists(model_path):
        print("❌ No trained model found! Run training (train) first.")
        return
    
    print(f"\n🔍 Analyzing: {file_path}")
    # Load the saved model
    clf = joblib.load(model_path)
    
    # Extract the pattern of the incoming audio file
    features = extract_features(file_path)
    if features is None:
        return
    
    # Run the prediction
    features = features.reshape(1, -1)
    prediction = clf.predict(features)[0]
    probabilities = clf.predict_proba(features)[0]
    
    # Print the result
    classes = {0: "Real human (Real)", 1: "AI-synthesized (Fake)"}
    result = classes[prediction]
    
    print("="*40)
    print(f"🚨 Result: [{result}]")
    print(f"📊 Probability -> Real: {probabilities[0]*100:.1f}%, Fake: {probabilities[1]*100:.1f}%")
    print("="*40)

# 4. Entry point
if __name__ == "__main__":
    # Step 1: train the model (control it by commenting/uncommenting)
    train_model(data_dir="data")
    
    # Step 2: run a prediction if a test file exists
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        predict_voice(test_file)
    else:
        print(f"💡 No '{test_file}' file to test, so the prediction is skipped.")
