import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv("cybercrime.csv")
data = data.dropna()

# Remove sensitive columns from ML
data = data.drop(["Name", "Card_Number"], axis=1)

encoders = {}

columns = [
    "City",
    "Crime_Type",
    "Time_of_Crime",
    "Victim_Age_Group",
    "Transaction_Mode",
    "Bank_Type",
    "Day_of_Week",
    "Location"
]

# Encode categorical columns
for col in columns:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    encoders[col] = encoder

# ML Features
X = data[
    [
        "City",
        "Crime_Type",
        "Amount",
        "Time_of_Crime",
        "Victim_Age_Group",
        "Transaction_Mode",
        "Bank_Type",
        "Day_of_Week",
        "Month",
        "Hour"
    ]
]

y = data["Location"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

print("Model Accuracy:", model.score(X_test, y_test))

# Save Model
joblib.dump(model, "cybercrime_model.pkl")

# Save Encoders
joblib.dump(encoders["City"], "city_encoder.pkl")
joblib.dump(encoders["Crime_Type"], "Crime_Type_encoder.pkl")
joblib.dump(encoders["Time_of_Crime"], "Time_of_Crime_encoder.pkl")
joblib.dump(encoders["Victim_Age_Group"], "Victim_Age_Group_encoder.pkl")
joblib.dump(encoders["Transaction_Mode"], "Transaction_Mode_encoder.pkl")
joblib.dump(encoders["Bank_Type"], "Bank_Type_encoder.pkl")
joblib.dump(encoders["Day_of_Week"], "Day_of_Week_encoder.pkl")
joblib.dump(encoders["Location"], "location_encoder.pkl")

print("Training Completed Successfully")
