import pickle
import librosa
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

SAMPLE_RATE = 16000  # Ensure 16kHz sample rate
MODEL_PATH = "cry_xgboost_model.pkl"

# Load the trained XGBoost model and class names
with open(MODEL_PATH, "rb") as model_file:
    model, class_names = pickle.load(model_file)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" in request.files:  # Handle file upload case (Optional)
        file = request.files["file"]
        audio_data = file.read()
    else:
        audio_data = request.data  # ESP32 sends raw bytes

    try:
        # Convert raw bytes to NumPy int16 array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        audio_array /= np.max(np.abs(audio_array))  # Normalize between -1 and 1

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio_array, sr=SAMPLE_RATE, n_mfcc=25)
        mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)  # Reshape for model input

        # **Make a prediction**
        predicted_label = model.predict(mfcc_mean)[0]
        predicted_condition = class_names[int(predicted_label)]  # Map to actual condition name

        return jsonify({"condition": predicted_condition})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
