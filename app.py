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
    try:
        data = request.get_json(force=True)

        # Validate required features
        missing = [f for f in selected_features if f not in data]
        if missing:
            return jsonify({
                "error": "Missing required features",
                "missing_fields": missing
            }), 400

        # Create DataFrame and ensure correct column order
        df = pd.DataFrame([data])
        df = df.reindex(columns=selected_features)

        # Scale and predict
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": round(probability, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run()
