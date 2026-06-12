```markdown
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

```

## ⚙️ How to Run
```bash
# 1. Clone the repository
git clone https://github.com/Yavar-NK/Churn-Prediction.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the training pipeline
python src/train.py

# 4. Build and run with Docker
docker build -t churn-prediction .
docker run -p 5000:5000 churn-prediction