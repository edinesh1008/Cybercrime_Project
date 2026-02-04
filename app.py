import streamlit as st
import joblib

# Load ML Model
model = joblib.load("cybercrime_model.pkl")

# Load Encoders
encoders = {
    "City": joblib.load("city_encoder.pkl"),
    "Crime_Type": joblib.load("Crime_Type_encoder.pkl"),
    "Time_of_Crime": joblib.load("Time_of_Crime_encoder.pkl"),
    "Victim_Age_Group": joblib.load("Victim_Age_Group_encoder.pkl"),
    "Transaction_Mode": joblib.load("Transaction_Mode_encoder.pkl"),
    "Bank_Type": joblib.load("Bank_Type_encoder.pkl"),
    "Day_of_Week": joblib.load("Day_of_Week_encoder.pkl"),
    "Location": joblib.load("location_encoder.pkl")
}

st.title("Cybercrime Prediction System")

# Date & Time Inputs
crime_date = st.date_input("Select Crime Date")
crime_time = st.time_input("Select Crime Time")

month = crime_date.month
hour = crime_time.hour

# User Inputs
inputs = {}

for col in list(encoders.keys())[:-1]:
    inputs[col] = st.selectbox(f"Select {col}", encoders[col].classes_)

# Amount Input
amount = st.number_input("Enter Fraud Amount", min_value=1)

# Prediction
if st.button("Predict"):

    if amount <= 0:
        st.warning("⚠ Please enter valid Fraud Amount")

    else:
        encoded_input = []

        for col in list(encoders.keys())[:-1]:
            encoded_input.append(encoders[col].transform([inputs[col]])[0])

        # Insert Amount
        encoded_input.insert(2, amount)

        # Add Month & Hour
        encoded_input.append(month)
        encoded_input.append(hour)

        prediction = model.predict([encoded_input])

        result = encoders["Location"].inverse_transform(prediction)

        st.success(f"Predicted Crime Location: {result[0]}")
