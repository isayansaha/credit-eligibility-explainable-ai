import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("german_credit2.csv")

# Select features
features = [
    "loan_amount",
    "loan_duration",
    "credit_history",
    "savings_account",
    "employment_duration",
    "installment_rate",
    "age",
    "previous_credits"
]

X = df[features]

# Map credit_risk to binary target
df["target"] = df["credit_risk"].apply(lambda x: 1 if x == 1 else 0)
y = df["target"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model & scaler
joblib.dump(model, "credit_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained and saved successfully.")
