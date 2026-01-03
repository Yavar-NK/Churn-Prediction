from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def run_smote_model(X_train, y_train, X_test, y_test):
    # Apply SMOTE to balance the classes
    smote = SMOTE(sampling_strategy='minority')
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    
    # Model definition and training
    model = LogisticRegression()
    model.fit(X_sm, y_sm)
    
    # Predictions and evaluation
    y_pred = model.predict(X_test)
    print("Model Performance Analysis (SMOTE):")
    print(classification_report(y_test, y_pred))
    
    return model
