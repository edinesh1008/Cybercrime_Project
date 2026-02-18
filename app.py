import streamlit as st
import joblib

# Load model
model = joblib.load("cybercrime_model.pkl")

# Load encoders
encoders = {
    "City": joblib.load("City_encoder.pkl"),
    "Crime_Type": joblib.load("Crime_Type_encoder.pkl"),
    "Time_of_Crime": joblib.load("Time_of_Crime_encoder.pkl"),
    "Victim_Age_Group": joblib.load("Victim_Age_Group_encoder.pkl"),
    "Transaction_Mode": joblib.load("Transaction_Mode_encoder.pkl"),
    "Bank_Type": joblib.load("Bank_Type_encoder.pkl"),
    "Day_of_Week": joblib.load("Day_of_Week_encoder.pkl"),
    "Location": joblib.load("Location_encoder.pkl")
}

st.title("Cybercrime Location Prediction System")

# User inputs
inputs = {}

for col in list(encoders.keys())[:-1]:
    inputs[col] = st.selectbox(col, encoders[col].classes_)

amount = st.number_input("Fraud Amount", min_value=1)
month = st.number_input("Month (1-12)", min_value=1, max_value=12)
hour = st.number_input("Hour (0-23)", min_value=0, max_value=23)

if st.button("Predict Location"):

    encoded_input = [
        encoders["City"].transform([inputs["City"]])[0],
        encoders["Crime_Type"].transform([inputs["Crime_Type"]])[0],
        amount,
        encoders["Time_of_Crime"].transform([inputs["Time_of_Crime"]])[0],
        encoders["Victim_Age_Group"].transform([inputs["Victim_Age_Group"]])[0],
        encoders["Transaction_Mode"].transform([inputs["Transaction_Mode"]])[0],
        encoders["Bank_Type"].transform([inputs["Bank_Type"]])[0],
        encoders["Day_of_Week"].transform([inputs["Day_of_Week"]])[0],
        month,
        hour
    ]

    prediction = model.predict([encoded_input])
    location = encoders["Location"].inverse_transform(prediction)

    st.success(f"Predicted Crime Location: {location[0]}")
