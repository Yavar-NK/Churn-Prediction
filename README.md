# Churn Prediction Pipeline

This project implements a modular, production-ready machine learning pipeline for customer churn prediction. The focus is on reproducibility, scalability, and lifecycle management, utilizing best practices in MLOps.

## 🚀 Project Overview
The pipeline addresses class imbalance using SMOTE and leverages Logistic Regression to predict customer churn. The structure is designed to facilitate easy deployment, monitoring, and model versioning.

## 🛠 Tech Stack
* **Language:** Python
* **Pipeline:** Modular `src/` architecture
* **Experiment Tracking:** MLflow
* **Containerization:** Docker
* **Techniques:** SMOTE (Synthetic Minority Over-sampling Technique) for handling class imbalance.

## 📂 Directory Structure
```text
.
├── models/             # Serialized model artifacts
├── mlruns/             # MLflow experiment tracking logs
├── src/                # Source code for training and inference
│   ├── app.py          # Application entry point
│   └── train.py        # Training pipeline script
├── Dockerfile          # Container configuration
└── requirements.txt    # Project dependencies

## ⚙️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Yavar-NK/Churn-Prediction.git
```
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the training pipeline
python src/train.py

# 4. Build and run with Docker
docker build -t churn-prediction .
docker run -p 5000:5000 churn-prediction
```