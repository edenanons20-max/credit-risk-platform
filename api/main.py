from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Chargement du modèle et du scaler
with open("/app/models/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("/app/models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI(title="Credit Risk API", version="1.0")

# Schéma des données d'entrée
class ClientData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_1: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

@app.get("/")
def root():
    return {"message": "Credit Risk API is running 🚀"}

@app.post("/predict")
def predict(client: ClientData):
    data = np.array([[
        client.LIMIT_BAL, client.SEX, client.EDUCATION, client.MARRIAGE,
        client.AGE, client.PAY_1, client.PAY_2, client.PAY_3,
        client.PAY_4, client.PAY_5, client.PAY_6,
        client.BILL_AMT1, client.BILL_AMT2, client.BILL_AMT3,
        client.BILL_AMT4, client.BILL_AMT5, client.BILL_AMT6,
        client.PAY_AMT1, client.PAY_AMT2, client.PAY_AMT3,
        client.PAY_AMT4, client.PAY_AMT5, client.PAY_AMT6
    ]])

    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    return {
        "default_prediction": int(prediction),
        "default_label": "⚠️ Risque de défaut" if prediction == 1 else "✅ Pas de défaut",
        "default_probability": round(float(probability), 4)
    }