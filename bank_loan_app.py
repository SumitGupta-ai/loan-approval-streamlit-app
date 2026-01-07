import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("Loan_Approval_RF.pkl")

THRESHOLD = 0.45

st.set_page_config(page_title="Loan Approval App", layout="centered")

st.title("🏦 Loan Approval Prediction")
st.write("Enter applicant details below:")

# -------- Input Fields --------
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)

person_gender = st.selectbox("Gender", ["male", "female"])

person_income = st.number_input("Annual Income", value=50000)

loan_amount = st.number_input("Loan Amount", value=100000)

loan_intent = st.selectbox(
    "Loan Intent",
    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
)

loan_int_rate = st.number_input("Interest Rate (%)", value=12.5)

loan_percent_income = st.number_input("Loan % of Income", value=0.2)

credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

previous_default = st.selectbox(
    "Previous Loan Default",
    ["Yes", "No"]
)

# Prediction 

if st.button("Predict Loan Approval"):
    input_data = pd.DataFrame([{
        "person_age": person_age,
        "person_gender": person_gender,
        "person_income": person_income,
        "loan_amnt": loan_amount,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_default
    }])

    prob = model.predict_proba(input_data)[0][1]
    prediction = "Approved ✅" if prob >= THRESHOLD else "Rejected ❌"

    st.subheader("Result")
    st.write(f"**Approval Probability:** {prob:.2f}")
    st.write(f"**Decision (Threshold {THRESHOLD}):** {prediction}")