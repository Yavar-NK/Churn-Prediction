import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import clean_data, scale_features
from model_utils import run_smote_model

df = pd.read_csv('../data/IT_customer_churn.csv')


df = clean_data(df)
df = scale_features(df)


X = df.drop('Churn', axis='columns')
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)


run_smote_model(X_train, y_train, X_test, y_test)