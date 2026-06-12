# 🚀 Customer Churn Prediction MLOps Pipeline

## 📝 Description
This project implements an end-to-end Machine Learning pipeline to predict customer churn. It focuses on robust data preprocessing, feature engineering, and MLOps best practices to ensure **reproducibility** and **experiment tracking**.

## 🛠️ Getting Started
Ready to explore the pipeline? Follow these simple steps to set it up locally:

```bash
git clone [https://github.com/Yavar-NK/churn-prediction.git](https://github.com/Yavar-NK/churn-prediction.git)
cd churn-prediction

````

### ✨ Key Capabilities
- 🧹 **Automated Preprocessing**: Handles missing values, performs label encoding, and applies Min-Max scaling for numerical features as implemented in the `src/train.py` module.
- ⚙️ **Feature Engineering**: Utilizes `pandas.get_dummies` for categorical variables and cleans datasets by removing redundant identifiers and mapping categorical strings to numerical values.
- 📈 **Experiment Tracking**: Integrated with **MLflow** to track metrics, parameters, and model versions automatically during training runs (`mlflow.start_run()`).
- 🐳 **Reproducible Environment**: Includes `conda.yaml` and a `Dockerfile` to ensure the project runs identically across different development and production environments.

## Project Structure
The project follows a clean, modular structure:

- **src/train.py**: Contains the core training and preprocessing pipeline.
- **src/app.py**: Deployment application script.
- **models/**: Directory for registered MLflow model artifacts.
- **conda.yaml**: Configuration for the virtual environment.
- **Dockerfile**: Configuration for containerizing the project.
- **requirements.txt**: List of necessary Python dependencies.

## 🚀 Technical Implementation

The preprocessing logic is designed to ensure data quality through the following steps:

- 🧹 **Data Cleaning**: Dropping redundant IDs and handling "No internet service" flags.
- 📉 **Feature Scaling**: Normalizing charges to improve the model's convergence rate.
- 📊 **Experiment Tracking**: By wrapping the training process in `mlflow.start_run()`, the pipeline systematically logs every iteration, allowing for granular analysis of model performance.

---
*Developed by Yavar* 👨‍💻
