
## 🚀 Interactive Demo
You can run the initial analysis and experimentation directly in your browser:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17Bq21Rmcr7G-iZmhjhdylL_-vyBCFiHO?usp=sharing)

# IT Customer Churn Prediction

This project implements a modular Machine Learning pipeline to predict customer churn. It demonstrates professional coding standards and advanced techniques for handling imbalanced datasets.

## 🚀 Project Overview
* **Objective**: Predict which customers are likely to leave the service.
* **Technique**: Used **SMOTE** (Synthetic Minority Over-sampling Technique) to balance data.
* **Architecture**: Modular design with separate modules for preprocessing, modeling, and execution.

## 📁 Project Structure
* `src/`: Core Python modules (`preprocessing.py`, `model_utils.py`, `train.py`).
* `data/`: Dataset storage (`IT_customer_churn.csv`).
* `Churn_Prediction.ipynb`: Initial data exploration and testing.

## 📊 Key Results
* Achieved an **F1-score of approximately 80%** for the churn class using SMOTE and Logistic Regression.
* Improved model fairness by addressing class imbalance.

## 🛠️ How to Use
1. Clone the repository.
2. Install dependencies: `pip install pandas scikit-learn imbalanced-learn`
3. Run the training script: `python src/train.py`
