# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F
from fastapi.middleware.cors import CORSMiddleware

MODEL_DIR = "./saved_distilbert_model"  # path where you saved trainer.save_model()

# Request/response models
class PredictRequest(BaseModel):
    text: str
    flight_number: Optional[str] = None
    date: Optional[str] = None

class PredictResponse(BaseModel):
    label: str
    score: float
    logits: Optional[List[float]] = None

# Load model + tokenizer once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

label_map = {0: "dissatisfied or neutral", 1: "satisfied"}

app = FastAPI(title="Airline Sentiment API")

# Allow Blazor dev server - in production restrict origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your front-end URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text
    if not text or text.strip() == "":
        raise HTTPException(status_code=400, detail="Empty text")
    # Tokenize
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()[0]
        probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
        class_id = int(np.argmax(probs))
        score = float(probs[class_id])
    return PredictResponse(label=label_map[class_id], score=score, logits=logits.tolist())
