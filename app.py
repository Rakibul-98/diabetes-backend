from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}}, supports_credentials=True)

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

default_values = {
    "Pregnancies": 0,
    "SkinThickness": 0,
    "CholCheck": 1,
    "BMI_y": 25.0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "Fruits": 1,
    "Veggies": 1,
    "GenHlth": 3, 
    "MentHlth": 0,
    "PhysHlth": 0,
    "Age_y": 40,
    "Education": 4,
    "Income": 3,
    "DiabetesPedigreeFunction": 0.0
}

@app.route('/')
def home():
    return "🚀 Diabetes Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        for feature in selected_features:
            if feature not in data:
                default = default_values.get(feature)
                if default is not None:
                    data[feature] = default

        missing = [f for f in selected_features if f not in data]
        if missing:
            return jsonify({
                "error": "Missing required features and no defaults available.",
                "missing_fields": missing
            }), 400

        df = pd.DataFrame([data])
        df = df.reindex(columns=selected_features)

        df_scaled = scaler.transform(df)
        prediction = int(model.predict(df_scaled)[0])
        probability = float(model.predict_proba(df_scaled)[0][1])

        return jsonify({
            "prediction": prediction,
            "probability": round(probability, 4),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
