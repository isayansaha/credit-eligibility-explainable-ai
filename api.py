from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model and scaler
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = [
    "loan_amount",
    "loan_duration",
    "credit_history",
    "savings_account",
    "employment_duration",
    "installment_rate",
    "age",
    "previous_credits"
]

@app.post("/predict")
def predict(data: dict):
    X = np.array([[data[f] for f in FEATURES]])
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0][1]

    if prob >= 0.7:
        risk = "Low"
    elif prob >= 0.4:
        risk = "Medium"
    else:
        risk = "High"

    importance = dict(zip(FEATURES, model.coef_[0]))

    top_factors = dict(
        sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    )

    return {
        "approval_probability": round(float(prob), 2),
        "risk_level": risk,
        "top_factors": top_factors
    }
