# 🧠 Diabetes Prediction Backend API

[![Node.js](https://img.shields.io/badge/Node.js-20.11.1-green)](https://nodejs.org/)
[![Express.js](https://img.shields.io/badge/Express-4.18.2-%23000000)](https://expressjs.com/)
[![Python](https://img.shields.io/badge/Python-3.11.8-blue)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Diabetes-orange)](https://scikit-learn.org/)

A powerful RESTful **API backend** built for **diabetes prediction** using machine learning and explainable AI. This project connects a Next.js frontend to a Python model for real-time predictions and insights based on medical input features.

➡ **Live Url:** [https://diabetes-client.vercel.app](https://diabetes-client.vercel.app)

## 🚀 Features

- **Prediction Endpoint** – Receives user input and returns diabetes risk predictions.
- **Explainable AI** – Integrated SHAP explanations to interpret model decisions.
- **Python Integration** – Uses child processes to communicate with a Python script.
- **CORS-Enabled API** – Supports secure communication with frontend apps.
- **Environment Configs** – Easily set API keys, ports, and paths.

## 🛠 Tech Stack

- **Backend:** Python, Flask
- **Model:** Python (scikit-learn, shap)
- **ML Algorithm:** Logistic Regression, XGBoost, SVM, Random Forest (Ensamble model)
- **Communication:** `child_process` for Node–Python interaction
- **Deployment Ready:** Deployed on Render.

## 📦 Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Rakibul-98/diabetes-backend.git
   cd diabetes-backend
   ```

2. **Install Node dependencies:**

   ```bash
   npm install
   ```

3. **Install Python dependencies:**

   Make sure Python 3.11+ is installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the root:

   ```env
   PORT=5000
   PYTHON_SCRIPT_PATH=./model/diabetes_predict.py
   ```

5. **Run the backend server:**

   ```bash
   npm run dev
   ```

6. **Test API in browser or Postman:**

   Visit: [http://localhost:5000/api/predict](http://localhost:5000/api/predict)

## 🔌 API Endpoints

### `POST /api/predict`

**Description:** Predicts diabetes probability based on input data.

**Request Body:**

```json
{
  "Glucose": 120,
  "BloodPressure": 70,
  "Insulin": 80,
  "BMI": 24.5,
  "Age": 45,
  "HighBP": 1,
  "HighChol": 1,
  "Smoker": 0,
  "PhysActivity": 1
}
```

**Response:**

```json
{
  "prediction": 1,
  "probability": 0.84,
  "explanation": { "Glucose": 0.31, "BMI": 0.22, ... }
}
```

## 📊 ML Model Details

- **Input Features:** Glucose, BloodPressure, Insulin, BMI, Age, HighBP, HighChol, Smoker, PhysActivity
- **Output:** Diabetes Prediction (0 = No, 1 = Yes)
- **Explainability:** SHAP values included in every response for transparency.

## 🔍 Future Enhancements

- JWT-based user authentication
- Save prediction history by user
- Upload CSV for batch predictions
- Admin dashboard for analytics
- Dockerize entire backend
- Real-time AI explanation visualization

## 📬 Contact

**Md Rakibul Hasan**

- Portfolio: [https://portfolio-rakibul.netlify.app](https://portfolio-rakibul.netlify.app)
- Email: [rakibul.rupom2001@gmail.com](mailto:rakibul.rupom2001@gmail.com)
