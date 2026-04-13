from fastapi import FastAPI
import joblib
import mlflow

app = FastAPI()

model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}

@app.post("/predict")
@mlflow.trace
def predict(data: dict):
    features = list(data.values())
    prediction = model.predict([features])
    return {"churn": int(prediction[0])}