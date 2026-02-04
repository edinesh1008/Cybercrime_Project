import streamlit as st
import joblib

model = joblib.load("cybercrime_model.pkl")
city_encoder = joblib.load("city_encoder.pkl")
crime_encoder = joblib.load("crime_encoder.pkl")
location_encoder = joblib.load("location_encoder.pkl")

st.title("Cybercrime Prediction System")

city = st.selectbox("Select City", city_encoder.classes_)
crime = st.selectbox("Select Crime Type", crime_encoder.classes_)
amount = st.number_input("Enter Fraud Amount")

if st.button("Predict"):

    city_encoded = city_encoder.transform([city])
    crime_encoded = crime_encoder.transform([crime])

    prediction = model.predict([[city_encoded[0], crime_encoded[0], amount]])

    location = location_encoder.inverse_transform(prediction)

    st.success(f"Predicted Crime Location: {location[0]}")
