import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import clean_data, scale_features
from model_utils import run_smote_model

# 1. Load the dataset
# Update the path based on your folder structure
try:
    df = pd.read_csv('../data/IT_customer_churn.csv')
except FileNotFoundError:
    url = "https://raw.githubusercontent.com/Yavar-NK/Churn-Prediction/main/IT_customer_churn.csv"
    df = pd.read_csv(url)

# 2. Data Cleaning and Scaling
df_cleaned = clean_data(df)
df_final = scale_features(df_cleaned)

# 3. Feature Selection
X = df_final.drop('Churn', axis='columns')
y = df_final['Churn']

# 4. Split Data (Stratified for better distribution)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)

# 5. Training and Evaluation
print("Executing ML Pipeline with SMOTE...")
model = run_smote_model(X_train, y_train, X_test, y_test)
print("Process Finished.")
