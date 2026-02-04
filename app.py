import streamlit as st
import joblib

model = joblib.load("cybercrime_model.pkl")

st.title("Cybercrime Prediction App")

city = st.number_input("Enter City Code")
crime = st.number_input("Enter Crime Type Code")
amount = st.number_input("Enter Amount")

if st.button("Predict"):
    result = model.predict([[city, crime, amount]])
    st.write("Predicted Location:", result)
