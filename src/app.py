import logging
import time
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
import traceback

# Setup logging configuration for monitoring purposes
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

app = FastAPI(title="IT Customer Churn Prediction Service")
MODEL_PATH = "models/churn_model"

# Load the machine learning model during application startup
@app.on_event("startup")
def load_model():
    global model
    try:
        model = mlflow.sklearn.load_model(MODEL_PATH)
        logging.info("Model loaded successfully!")
    except Exception as e:
        logging.error(f"Failed to load model. Error: {e}")
        model = None

# Middleware to log request processing time (Latency Monitoring)
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logging.info(f"Path: {request.url.path} | Processing Time: {process_time:.4f}s")
    return response

# Define the input data schema for customer information
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    PaperlessBilling: int
    MonthlyCharges: float
    TotalCharges: float
    InternetService_DSL: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    Contract_Month_to_month: int
    Contract_One_year: int
    Contract_Two_year: int
    PaymentMethod_Bank_transfer: int
    PaymentMethod_Credit_card: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

# Prediction endpoint with comprehensive error handling
@app.post("/predict")
def predict_churn(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Log incoming prediction request
        logging.info(f"Predicting for input data: {data.model_dump()}")
        
        # Convert Pydantic model to a pandas DataFrame
        input_dict = data.model_dump()
        input_data = pd.DataFrame([input_dict])
        
        # 🎯 Apply feature scaling to match the training pipeline distribution (MinMaxScaler simulation)
        input_data['tenure'] = input_data['tenure'] / 72.0
        input_data['MonthlyCharges'] = (input_data['MonthlyCharges'] - 18.25) / (118.75 - 18.25)
        input_data['TotalCharges'] = (input_data['TotalCharges'] - 18.8) / (8684.8 - 18.8)
        
        # Align features with the model's training expectations to prevent mismatches
        expected_features = model.feature_names_in_
        input_data = input_data.reindex(columns=expected_features, fill_value=0)
        
        # Perform prediction and retrieve probabilities
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]
        
        result = {
            "churn_prediction": int(prediction[0]),
            "churn_probability": float(probability)
        }
        
        # Log the prediction result
        logging.info(f"Prediction result: {result}")
        return result
        
    except Exception as e:
        # Log detailed traceback in case of failure
        logging.error(f"Prediction Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)