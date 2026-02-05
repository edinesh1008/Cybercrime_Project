# ===============================
# IMPORTS
# ===============================
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


# ===============================
# HELPER FUNCTIONS
# ===============================

# Transaction Timeline
def show_transaction_timeline():
    times = ["09:00","11:30","13:00","15:45","18:20"]
    amounts = [5000,12000,8000,45000,20000]

    df = pd.DataFrame({"Time":times,"Amount":amounts})

    fig, ax = plt.subplots()
    ax.plot(df["Time"],df["Amount"],marker="o")
    st.pyplot(fig)


# Fraud Network Graph
def show_fraud_network():
    G = nx.Graph()
    G.add_edges_from([
        ("Victim","Account A"),
        ("Account A","Account B"),
        ("Account B","Account C")
    ])

    fig, ax = plt.subplots()
    nx.draw(G,with_labels=True,node_color="lightblue",ax=ax)
    st.pyplot(fig)


# Risk Score
def show_risk_score(amount):
    score = min(amount/1000,100)
    st.progress(int(score))
    st.write(f"Risk Score: {int(score)}%")


# Bank Statement
def generate_bank_statement():
    data = {
        "Transaction":["UPI Transfer","ATM Withdrawal","Online Purchase"],
        "Amount":[45000,5000,2000]
    }
    st.table(pd.DataFrame(data))


# Complaint PDF
def generate_pdf(name, location):
    file = "complaint.pdf"
    doc = SimpleDocTemplate(file)
    styles = getSampleStyleSheet()

    content = [
        Paragraph(f"Victim Name: {name}",styles["Normal"]),
        Paragraph(f"Crime Location: {location}",styles["Normal"])
    ]

    doc.build(content)
    return file


# ===============================
# LOAD MODEL & ENCODERS
# ===============================
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


# ===============================
# STREAMLIT UI
# ===============================
st.title("Cybercrime Investigation Dashboard")

name = st.text_input("Victim Name")
amount = st.number_input("Fraud Amount",min_value=1)

inputs = {}

for col in list(encoders.keys())[:-1]:
    inputs[col] = st.selectbox(col, encoders[col].classes_)


# ===============================
# PREDICTION
# ===============================
if st.button("Predict"):

    encoded_input = []

    for col in list(encoders.keys())[:-1]:
        encoded_input.append(encoders[col].transform([inputs[col]])[0])

    encoded_input.insert(2, amount)

    prediction = model.predict([encoded_input])
    result = encoders["Location"].inverse_transform(prediction)

    st.success(f"Predicted Location: {result[0]}")


    # ===============================
    # VISUALIZATION + REPORTS
    # ===============================

    show_transaction_timeline()
    show_fraud_network()
    show_risk_score(amount)
    generate_bank_statement()

    pdf = generate_pdf(name,result[0])

    with open(pdf,"rb") as f:
        st.download_button("Download Complaint PDF",f,file_name=pdf)
