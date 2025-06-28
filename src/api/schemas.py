from pydantic import BaseModel, Field
from typing import List, Optional

class CustomerFeatures(BaseModel):
    """
    Input features for customer risk prediction.
    """
    recency: float = Field(..., description="Days since last transaction")
    frequency: int = Field(..., description="Number of transactions")
    monetary: float = Field(..., description="Total transaction value")
    currency_code: str = Field(..., description="Currency code")
    country_code: str = Field(..., description="Country code")
    product_categories: List[str] = Field(..., description="List of product categories")
    avg_transaction_amount: float = Field(..., description="Average transaction amount")
    transaction_std: float = Field(..., description="Standard deviation of transaction amounts")

class PredictionResponse(BaseModel):
    """
    Response model for prediction endpoint.
    """
    customer_id: str = Field(..., description="Customer identifier")
    risk_probability: float = Field(..., description="Predicted risk probability", ge=0, le=1)
    risk_score: int = Field(..., description="Credit risk score (0-100)")
    recommended_credit_limit: float = Field(..., description="Recommended credit limit")
    model_version: str = Field(..., description="Model version used for prediction")

class HealthCheckResponse(BaseModel):
    """
    Response model for health check endpoint.
    """
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
