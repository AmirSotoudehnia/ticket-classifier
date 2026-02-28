from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=5000)

class PredictResponse(BaseModel):
    label: str
    confidence: float