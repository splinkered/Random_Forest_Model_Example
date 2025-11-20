import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pickle
import pandas as pd

BASE_DIR = os.path.dirname(__file__)  # points to api/
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
static_path = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
def home():
    return FileResponse(os.path.join(static_path, "index.html"))

# Load model
model = pickle.load(open(os.path.join(BASE_DIR, "heart_model.pkl"), "rb"))
label_encoder = pickle.load(open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb"))

# Precomputed metrics
metrics_data = {
    "accuracy": 0.9811,
    "report": """precision    recall  f1-score   support
0       0.98      0.97      0.98       101
1       0.98      0.99      0.98       163
accuracy                           0.98       264
macro avg       0.98      0.98      0.98       264
weighted avg       0.98      0.98      0.98       264""",
    "confusion_matrix_image": os.path.join(static_path, "metrics/confusion_matrix.png")
}

@app.get("/metrics")
def get_metrics():
    return metrics_data

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred_num = model.predict(df)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]
    return {"prediction": str(pred_label)}
