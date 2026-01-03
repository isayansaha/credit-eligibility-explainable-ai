from fastapi import FastAPI
import requests
import google.generativeai as genai

genai.configure(api_key="AIzaSyC24D_AiCIOPFZyfUDWh5NvmwKle6s-pPY")

app = FastAPI()

ML_API_URL = "http://127.0.0.1:8000/predict"
model = genai.GenerativeModel("models/gemini-2.5-flash")

def build_prompt(ml_output, loan_amount):
    return f"""
You are a financial decision-support assistant.
You do NOT approve or reject loans.
You ONLY explain ML outputs and provide educational suggestions.

Loan amount requested: {loan_amount}
Approval probability: {ml_output['approval_probability'] * 100:.0f}%
Risk level: {ml_output['risk_level']}
Top contributing factors: {ml_output['top_factors']}

Explain:
1. Why this risk level was assigned.
2. 3 practical suggestions to improve eligibility.
Add a one-line disclaimer that this is educational.
"""

@app.post("/analyze-loan")
def analyze_loan(data: dict):
    ml_response = requests.post(ML_API_URL, json=data)
    ml_output = ml_response.json()

    prompt = build_prompt(ml_output, data["loan_amount"])
    gemini_response = model.generate_content(prompt)

    return {
        "loan_analysis": ml_output,
        "explanation": gemini_response.text
    }
