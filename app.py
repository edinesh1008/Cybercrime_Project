import streamlit as st
import joblib
import networkx as nx
import matplotlib.pyplot as plt

# ================= LOAD MODEL =================
model = joblib.load("cybercrime_model.pkl")

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

st.title("Cybercrime Investigation Dashboard")

# ================= USER DETAILS =================
name = st.text_input("Victim Name")
card_number = st.text_input("Card Number (12 digits only)")

# ================= DATE & TIME =================
crime_date = st.date_input("Select Crime Date")
crime_time = st.time_input("Select Crime Time")

month = crime_date.month
hour = crime_time.hour

# ================= CRIME INPUT =================
inputs = {}

for col in list(encoders.keys())[:-1]:
    inputs[col] = st.selectbox(col, encoders[col].classes_)

amount = st.number_input("Fraud Amount", min_value=1)

# ================= PREDICTION =================
if st.button("Predict"):

    if name.strip() == "":
        st.warning("Enter victim name")

    elif not card_number.isdigit() or len(card_number) != 12:
        st.warning("Card number must be 12 digits")

    else:

        encoded_input = [
            encoders["City"].transform([inputs["City"]])[0],
            encoders["Crime_Type"].transform([inputs["Crime_Type"]])[0],
            amount,
            encoders["Time_of_Crime"].transform([inputs["Time_ofC_Time"] if False else inputs["Time_of_Crime"]])[0],
            encoders["Victim_Age_Group"].transform([inputs["Victim_Age_Group"]])[0],
            encoders["Transaction_Mode"].transform([inputs["Transaction_Mode"]])[0],
            encoders["Bank_Type"].transform([inputs["Bank_Type"]])[0],
            encoders["Day_of_Week"].transform([inputs["Day_of_Week"]])[0],
            month,
            hour
        ]

        prediction = model.predict([encoded_input])
        result = encoders["Location"].inverse_transform(prediction)

        st.success(f"Predicted Crime Location: {result[0]}")

        # ================= FRAUD NETWORK =================
        st.subheader("Fraud Network")

        G = nx.Graph()
        G.add_edges_from([
            ("Victim","Account A"),
            ("Account A","Account B"),
            ("Account B","Account C")
        ])

        fig, ax = plt.subplots()
        nx.draw(G, with_labels=True, node_color="lightblue", ax=ax)
        st.pyplot(fig)

        # ================= RISK SCORE =================
        st.subheader("Risk Score")

        score = min(amount/1000, 100)
        st.progress(int(score))
        st.write(f"Risk Level: {int(score)}%")
