# Bankruptcy Prediction API

This project provides a **Machine Learning API** for predicting whether a company is at risk of bankruptcy, using financial metrics. It uses a logistic regression model trained on 15 top-correlated features from real-world bankruptcy data.

---

## 👥 Who Are the Users?

| User Type        | Description                                                              |
|------------------|---------------------------------------------------------------------------|
| **Financial Analysts** | Evaluate risk of bankruptcy for companies in real-time or batch |
| **Lending Institutions** | Integrate with API before approving loans or investment    |
| **Risk Modeling Teams** | Monitor bankruptcy risk over time using batch uploads       |

---

## ⚙️ Setup Instructions

### ✅ 1. Clone or download this repo

```bash
git clone https://github.com/your-username/bankruptcy_api.git
cd bankruptcy_api

### ✅ 2. Train the model
python train_model.py

### ✅ 3. Run the API locally (Docker Desktop)
docker build -t bankruptcy-api .
docker run -d -p 8080:8080 bankruptcy-api

Visit the API in browser: http://localhost:8080/docs

Cloud Deployment (Google Cloud Run)
🔧 Build and deploy:
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/bankruptcy-api
gcloud run deploy bankruptcy-api \
  --image gcr.io/YOUR_PROJECT_ID/bankruptcy-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

✅ Live API URL

📦 Tech Stack
Python 3.10

FastAPI (for the API)

scikit-learn (ML model)

SHAP (explainability)

Docker (containerization)

Google Cloud Run (deployment)

Joblib (model saving/loading)

🔌 API Endpoint
POST /predict
✅ Sample Request:
{
  "features": [0.23, 0.41, 0.67, 0.35, 0.28, 0.14, 0.55, 0.73, 0.12, 0.66, 0.33, 0.89, 0.37, 0.45, 0.25]
}

✅ Sample Response:
{
  "prediction": 1,
  "probability": 0.8421
}

📊 Evaluation Metrics
✅ F1 Score

✅ Precision & Recall

✅ ROC AUC

✅ Average Precision Score

✅ Brier Score

📈 How It Works: System Diagram
+------------+         HTTP POST           +--------------------------+
|   User     |  -------------------------> |  FastAPI ML API          |
| (Frontend) |                             |  /predict endpoint       |
+------------+                             |  Model: LogisticRegression
                                           |  Scaler: StandardScaler  |
                                           +------------+-------------+
                                                        |
                                                        v
                                          Returns prediction & probability

📁 Project Structure
bankruptcy_api/
├── app/
│   ├── main.py           # FastAPI app entry point
│   ├── model.py          # Model loader + predict() function
│   └── schema.py         # Pydantic request/response models
├── model/
│   ├── bankruptcy_model.joblib
│   ├── scaler.joblib
│   └── feature_names.joblib
├── train_model.py        # Model training script
├── Dockerfile            # Docker build instructions
├── requirements.txt      # Python dependencies
└── README.md             # This file

👩‍💻 Author
Genae Adams
Assignment 2 — Designing and Deploying a Machine Learning API
Systems Analysis & Design, 2025
