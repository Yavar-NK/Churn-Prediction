# 🚀 Customer Churn Prediction MLOps Pipeline

## 📝 Description
This project implements an end-to-end Machine Learning pipeline to predict customer churn. It focuses on robust data preprocessing, feature engineering, and MLOps best practices to ensure **reproducibility**, **experiment tracking**, and **production-ready model serving**.

## 🛠️ Getting Started
Follow these simple steps to set up and run the pipeline locally, or quickly explore the workflow in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SVJtXVcO3YsvniMFE6wkm-ocE2iws4HA)

### 1. Clone the Repository & Navigate
```bash
git clone https://github.com/Yavar-NK/Churn-Prediction.git
cd Churn-Prediction
```

## 🚀 Local Execution Guide

Follow these step-by-step instructions to set up the environment, train the model, and deploy the production API locally.

### 1️⃣ Install Dependencies
Ensure you have Python installed, then install all the required MLOps and Machine Learning packages using `pip`:
```bash
pip install -r requirements.txt
```
### 2️⃣ Run Training Pipeline (MLflow Tracking)
Make sure the dataset IT_customer_churn.csv is placed directly in the root directory. Then, execute the pipeline to handle data balancing (SMOTE), model training, and MLflow logging : 
python src/train.py 

💡 Note: This will automatically log your metrics, parameters, and model artifacts into the local mlruns directory.

### 3️⃣ Start FastAPI Production Server
Once the model is successfully trained and saved under the models/ directory, launch the high-performance Uvicorn web 'server':
```bash 
uvicorn src.app:app --reload
```

### 🔍 Interactive API Testing
After starting the server, open your browser and navigate 'to':
```bash

👉 http://127.0.0.1:8000/docs
```
This will open the interactive Swagger UI documentation, allowing you to send live test payloads (JSON requests) to the model and receive real-time churn predictions.


### ✨ Key Capabilities

| Capability | Tech Stack | Engineered Implementation |
| :--- | :--- | :--- |
| **🧹 Automated Preprocessing** | `pandas` \| `scikit-learn` | Dynamic missing value removal, precise binary mapping, and identical column-alignment across pipelines. |
| **📉 Robust Feature Scaling** | `MinMaxScaler` Simulation | Synchronized feature range normalization ($0$ to $1$) hardcoded inside the inference layer to completely eliminate data skew. |
| **⚖️ Class Imbalance Handling** | `SMOTE` (Imblearn) | Synthetic over-sampling of the minority class to drastically optimize model `recall_score` on churners. |
| **📈 Experiment Tracking** | `MLflow` | Automatic logging of parameters (`max_iter`, `C`), performance metrics (`accuracy`, `f1_score`), and granular model versioning. |
| **⚡ Production Serving** | `FastAPI` \| `Pydantic` | High-performance async REST API with rigorous request schema validation and real-time latency tracking middleware. |
| **🐳 Container Deployment** | `Docker` | Isolated, immutable, and 100% reproducible execution environment via optimized `Dockerfile`. |

> 💡 **Architectural Note:** The preprocessing and feature scaling distributions are mathematically aligned between `train.py` and `app.py`. This ensures that real-time API payloads undergo the exact same transformation as the training data, ensuring production stability.

### 📁 Project Structure

The repository maintains a production-grade, modular layout separating the core training pipeline from the inference serving layer:

```text
Churn-Prediction/
├── models/
│   └── churn_model/         # Serialized MLflow model artifacts (logs, weights, meta)
├── src/
│   ├── app.py               # High-performance FastAPI production serving script
│   └── train.py             # Core pipeline (Preprocessing, SMOTE, MLflow tracking)
├── Dockerfile               # Optimized deployment containerization recipe
├── IT_customer_churn.csv    # Local dataset ensuring 100% network-independent execution
└── requirements.txt         # Hardcoded python dependencies for environment lock-in
```

| Component | Responsibility | Key Engineering Focus |
| :--- | :--- | :--- |
| 🛠️ `src/train.py` | Pipeline Automation | Ingests local CSV, runs SMOTE, and registers artifacts to MLflow registry. |
| ⚡ `src/app.py` | Production Serving | Initializes FastAPI web-server, applies real-time scaling, and exposes `/predict`. |
| 📦 `models/` | Artifact Storage | Holds the frozen operational model state used live by the inference layer. |
| 🐳 `Dockerfile` | Containerization | Packages the app into an isolated, immutable layer ready for AWS/GCP or local Docker run. |

## 🚀 Technical Implementation

> [!IMPORTANT]
> ### 📊 Class Imbalance Mitigation (SMOTE)
> To prevent the Logistic Regression model from biasing toward non-churning customers, the training pipeline synthesizes new minority class examples using **SMOTE**. This guarantees balanced decision boundaries and dramatically improves the model's sensitivity (`recall_score`) toward high-risk accounts.

> [!TIP]
> ### 📉 Production Feature Alignment & Scaling
> Real-time input JSON payloads submitted to the FastAPI `/predict` endpoint undergo dynamic normalization. By simulating the `MinMaxScaler` distribution parameters inside the serving layer, the system completely eliminates **Data Skew / Feature Mismatch** (preventing artificial $1.0$ probability locks).

> [!NOTE]
> ### 🔬 Granular Experiment Auditing (MLflow Registry)
> By wrapping the training execution inside `mlflow.start_run()`, every hyperparameter (`C`, `max_iter`) and evaluation metric is systematically audited. The frozen model state is then seamlessly registered under the `models/churn_model` directory for immutable production deployment.




