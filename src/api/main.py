from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import numpy as np
import logging
from datetime import datetime
from .schemas import CustomerFeatures, PredictionResponse, HealthCheckResponse

app = FastAPI(title="Credit Risk Prediction API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
try:
    model = mlflow.sklearn.load_model("models:/credit_risk_model/Production")
    logger.info("Successfully loaded model")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise Exception("Failed to load model")

def calculate_credit_score(probability: float) -> int:
    """
    Convert risk probability to credit score (0-100).
    """
    # Higher probability means higher risk, so lower score
    return int((1 - probability) * 100)

def calculate_credit_limit(probability: float, avg_transaction: float) -> float:
    """
    Calculate recommended credit limit based on risk probability.
    """
    # Base limit based on average transaction
    base_limit = avg_transaction * 10
    
    # Adjust based on risk probability
    risk_factor = 1 - probability
    return round(base_limit * risk_factor, 2)

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint to verify service status.
    """
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(customer_data: CustomerFeatures):
    """
    Predict credit risk probability for a customer.
    """
    try:
        # Prepare features for prediction
        features = {
            'Recency': customer_data.recency,
            'Frequency': customer_data.frequency,
            'Monetary': customer_data.monetary,
            'avg_transaction_amount': customer_data.avg_transaction_amount,
            'transaction_std': customer_data.transaction_std
        }
        
        # Convert to numpy array
        X = np.array([list(features.values())])
        
        # Get prediction
        risk_probability = model.predict_proba(X)[0][1]
        
        # Calculate derived metrics
        risk_score = calculate_credit_score(risk_probability)
        credit_limit = calculate_credit_limit(risk_probability, customer_data.avg_transaction_amount)
        
        return PredictionResponse(
            customer_id="",  # Placeholder for customer ID
            risk_probability=float(risk_probability),
            risk_score=risk_score,
            recommended_credit_limit=credit_limit,
            model_version="1.0.0"  # Model version placeholder
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
