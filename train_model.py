import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("cybercrime.csv")

# Columns
categorical_cols = [
    "City",
    "Crime_Type",
    "Time_of_Crime",
    "Victim_Age_Group",
    "Transaction_Mode",
    "Bank_Type",
    "Day_of_Week",
    "Location"
]

encoders = {}

# Encode categorical columns
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
    joblib.dump(le, f"{col}_encoder.pkl")

# Features & Target
X = data.drop("Location", axis=1)
y = data["Location"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "cybercrime_model.pkl")

print("Training Completed Successfully")
