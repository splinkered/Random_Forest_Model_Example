from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pickle
import pandas as pd
from metrics import generate_metrics

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

df = pd.read_csv("Medicaldataset.csv")
X = df.drop("Result", axis=1)
y = df["Result"]
y_encoded = label_encoder.transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Generate metrics
metrics_data = generate_metrics(model, X_test, y_test)

@app.get("/metrics")
def get_metrics():
    return {
        "accuracy": metrics_data["accuracy"],
        "report": metrics_data["report"],
        "confusion_matrix_image": metrics_data["cm_image"]
    }

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred_num = model.predict(df)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]
    return {"prediction": str(pred_label)}
