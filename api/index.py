from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pickle
import pandas as pd

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

# Load model and data
model = pickle.load(open("heart_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))


@app.get("/metrics")
def get_metrics():
    return {
        "accuracy": 0.9811,
        "report": """              precision    recall  f1-score   support

           0       0.98      0.97      0.98       101
           1       0.98      0.99      0.98       163

    accuracy                           0.98       264
   macro avg       0.98      0.98      0.98       264
weighted avg       0.98      0.98      0.98       264""",
        "confusion_matrix_image": "./static/metrics/confusion_matrix.png"
    }

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred_num = model.predict(df)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]
    return {"prediction": str(pred_label)}
