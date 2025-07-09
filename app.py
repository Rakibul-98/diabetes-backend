from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

@app.route('/')
def home():
    return "Diabetes Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])[selected_features]
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]
    return jsonify({
        "prediction": int(prediction),
        "probability": round(probability, 4)
    })

if __name__ == '__main__':
    app.run()
