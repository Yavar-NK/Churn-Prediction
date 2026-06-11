# Churn Prediction Pipeline

*The graph above illustrates the model's performance in identifying customer churn patterns.*

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** TensorFlow/Keras, Pandas, NumPy, Scikit-learn, MLflow
- **Environment:** Docker, Linux, Local/Cloud

## 🚀 Features
- Automated data preprocessing and feature engineering.
- Robust training pipeline with experiment tracking via MLflow.
- Modular architecture for seamless inference.
- Containerized deployment using Docker.

## 📁 Project Structure

```text
.
├── models/             # Serialized model artifacts
├── mlruns/             # MLflow experiment tracking logs
├── src/                # Source code for training and inference
│   ├── app.py          # Application entry point
│   └── train.py        # Training pipeline script
├── Dockerfile          # Container configuration
└── requirements.txt    # Project dependencies
