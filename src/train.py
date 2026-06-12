import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')

def preprocess_data(df):
    df = df[df["TotalCharges"] != " "]
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    if 'customerID' in df.columns:
        df.drop('customerID', axis='columns', inplace=True)
    replace_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in replace_cols:
        df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})
    yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                   'PaperlessBilling', 'Churn']
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].replace({'Yes': 1, 'No': 0})
    df['gender'] = df['gender'].replace({'Female': 1, 'Male': 0})
    df2 = pd.get_dummies(data=df, columns=['InternetService', 'Contract', 'PaymentMethod'], dtype=int)
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = MinMaxScaler()
    df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
    return df2

def train_pipeline():
    mlflow.set_experiment("IT_Customer_Churn_MLOps")
    with mlflow.start_run() as run:
        print(f"Successfully started MLflow Run ID: {run.info.run_id}")
        url = "https://raw.githubusercontent.com/Yavar-NK/Churn-Prediction/main/data/IT_customer_churn.csv"
        print("Fetching dataset from GitHub...")
        raw_df = pd.read_csv(url)
        print("Preprocessing dataset...")
        processed_df = preprocess_data(raw_df)
        X = processed_df.drop('Churn', axis='columns')
        y = processed_df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)
        print("Applying SMOTE balancing technique...")
        smote = SMOTE(random_state=15)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        params = {"max_iter": 1000, "C": 1.0, "random_state": 15}
        print("Training Logistic Regression model...")
        model = LogisticRegression(**params)
        model.fit(X_train_res, y_train_res)
        
        if not os.path.exists('models'):
            os.makedirs('models')
            
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall_score", rec)
        mlflow.log_metric("precision_score", prec)
        mlflow.log_metric("f1_score", f1)
        
        print("Saving and registering model...")
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", registered_model_name="IT_Logistic_Churn_Model")
        mlflow.sklearn.save_model(model, "models/churn_model")
        print("\n========================================")
        print("Phase 1 completed successfully!")
        print("========================================")

if __name__ == "__main__":
    train_pipeline()