import streamlit as st
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Cybercrime Intelligence System",
    page_icon="🛡️",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
    <style>
    .main-title {
        font-size:40px;
        font-weight:700;
        color:#1f4e79;
    }
    .sub-title {
        font-size:18px;
        color:gray;
    }
    .prediction-box {
        padding:20px;
        border-radius:12px;
        background-color:#f0f6ff;
        border:1px solid #d6e4ff;
        font-size:22px;
        font-weight:600;
        color:#0b3d91;
    }
    </style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = joblib.load("cybercrime_model.pkl")

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

# ================= HEADER =================
st.markdown('<div class="main-title">🛡 Cybercrime Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-based Prediction of Likely Fraud Withdrawal Location</div>', unsafe_allow_html=True)
st.markdown("---")

# ================= SIDEBAR INPUTS =================
st.sidebar.header("Enter Crime Details")

inputs = {}

for col in list(encoders.keys())[:-1]:
    inputs[col] = st.sidebar.selectbox(col.replace("_", " "), encoders[col].classes_)

amount = st.sidebar.number_input("Fraud Amount", min_value=1)
month = st.sidebar.number_input("Month (1-12)", min_value=1, max_value=12)
hour = st.sidebar.number_input("Hour (0-23)", min_value=0, max_value=23)

predict_button = st.sidebar.button("🔍 Predict Location")

# ================= MAIN PANEL =================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Case Summary")
    st.write("This system analyzes fraud complaint parameters and predicts the most probable withdrawal hotspot using a trained Machine Learning model.")

with col2:
    st.info("Model Used: Random Forest Classifier")

# ================= PREDICTION =================
if predict_button:

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

    st.markdown("---")
    st.markdown(
        f'<div class="prediction-box">📍 Predicted Crime Location: {location[0]}</div>',
        unsafe_allow_html=True
    )

    st.success("Prediction generated successfully using trained ML model.")

# ================= FOOTER =================
st.markdown("---")
st.caption("Cybercrime Intelligence System | Developed using Python, Scikit-Learn & Streamlit")
