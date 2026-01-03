import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data(df):
    # Handle missing values in TotalCharges
    df = df[df["TotalCharges"] != " "]
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    
    if 'customerID' in df.columns:
        df.drop('customerID', axis='columns', inplace=True)
        
    # Binary encoding for categorical columns
    yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity',
                      'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
    for col in yes_no_columns:
        df[col].replace({'Yes': 1, 'No': 0}, inplace=True)
        
    df['gender'].replace({'Male': 1, 'Female': 0}, inplace=True)
    
    # One-hot encoding for multi-class categories
    df = pd.get_dummies(df, columns=['Contract','PaymentMethod','InternetService'])
    return df

def scale_features(df):
    scaler = MinMaxScaler()
    cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df
