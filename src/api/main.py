from fastapi import FastAPI, HTTPException
from src.api.schemas import PredictRequest, PredictResponse
from src.models.predict import TicketClassifier
from src.utils.config import get_model_path, get_vectorizer_path

app = FastAPI(title="Ticket Classifier API", version="1.0.0")

classifier = None

@app.on_event("startup")
def load_model():
    global classifier
    if not get_model_path().exists() or not get_vectorizer_path().exists():
        classifier = None
        return
    classifier = TicketClassifier()

@app.get("/health")
def health():
    ready = classifier is not None
    return {"status": "ok", "model_loaded": ready}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    pred = classifier.predict_one(req.text)
    return PredictResponse(label=pred.label, confidence=pred.confidence)