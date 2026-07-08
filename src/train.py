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
    # Remove empty spaces in TotalCharges
    df = df[df["TotalCharges"] != " "]
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    
    # Drop redundant customerID column
    if 'customerID' in df.columns:
        df.drop('customerID', axis='columns', inplace=True)
        
    # Standardize 'No internet service' and 'No phone service' flags
    replace_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in replace_cols:
        df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})
        
    # Convert Yes/No categorical values to binary 1/0 values
    yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                   'PaperlessBilling', 'Churn']
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].replace({'Yes': 1, 'No': 0})
            
    # Encode gender column
    df['gender'] = df['gender'].replace({'Female': 1, 'Male': 0})
    
    # One-hot encode multi-class categorical features
    df2 = pd.get_dummies(data=df, columns=['InternetService', 'Contract', 'PaymentMethod'], dtype=int)
    
    # Scale numerical features using MinMaxScaler
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = MinMaxScaler()
    df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
    return df2

def train_pipeline():
    # Configure MLflow experiment tracking
    mlflow.set_experiment("IT_Customer_Churn_MLOps")
    with mlflow.start_run() as run:
        print(f"Successfully started MLflow Run ID: {run.info.run_id}")
        
        # Load local dataset to prevent network issues or API limits
        url = "IT_customer_churn.csv"
        print(f"Loading local dataset from: {url}")
        raw_df = pd.read_csv(url)
        
        print("Preprocessing dataset...")
        processed_df = preprocess_data(raw_df)
        X = processed_df.drop('Churn', axis='columns')
        y = processed_df['Churn']
        
        # Split dataset into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)
        
        # Apply SMOTE to handle class imbalance in the training data
        print("Applying SMOTE balancing technique...")
        smote = SMOTE(random_state=15)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        # Define model hyperparameters
        params = {"max_iter": 1000, "C": 1.0, "random_state": 15}
        
        print("Training Logistic Regression model...")
        model = LogisticRegression(**params)
        model.fit(X_train_res, y_train_res)
        
        # Ensure models directory exists
        if not os.path.exists('models'):
            os.makedirs('models')
            
        # Clean up existing model artifacts to prevent conflicts
        if os.path.exists('models/churn_model'):
            import shutil
            shutil.rmtree('models/churn_model')
            
        # Evaluate model performance
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log parameters and classification metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall_score", rec)
        mlflow.log_metric("precision_score", prec)
        mlflow.log_metric("f1_score", f1)
        
        # Log and save model artifacts
        print("Saving and registering model...")
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", registered_model_name="IT_Logistic_Churn_Model")
        mlflow.sklearn.save_model(model, "models/churn_model")
        
        print("\n========================================")
        print("Phase 1 completed successfully!")
        print("========================================")

if __name__ == "__main__":
    train_pipeline()