# ------------------------------------------------------------
# Loan Approval Predictor
# Author: Luca Franceschi
# Email: luca.franceschi01@estudiant.upf.edu
# All contents are created in-house unless explicitly stated.

# UI Created with assistance from ChatGPT (GPT-5, OpenAI)
# ------------------------------------------------------------

import streamlit as st
from utils import load_model, predict_sample_data
import numpy as np

model_data = load_model('loan_app_artifacts.pkl')

# Page configuration
st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

# Title and description
st.title("Loan Approval Predictor $$")
st.write("Enter your details to predict your personal loan approval probability.")

# Applicant Details Section
st.subheader("Applicant Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    applicant_income = st.number_input("Applicant Income ($)", min_value=150, step=100, max_value=81000, value=6000)
    coapplicant_income = st.number_input("Co-Applicant Income ($)", min_value=0, step=100, max_value=41700, value=8000)

with col2:
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    credit_history = st.selectbox("Credit History Available?", ["Yes (1.0)", "No (0.0)"])
    loan_amount = st.number_input("Loan Amount ($k)", min_value=9, step=1, max_value=700, value=150)
    loan_term = st.number_input("Loan Term (Months)", min_value=12, step=1, max_value=480, value=360)

# Prediction Cutoff Section
st.subheader("Prediction Cutoff")
st.write("Set the probability cutoff for approval")
cutoff = st.slider("", 0.0, 1.0, 0.50, 0.01)

# Predict Button
st.write("")
predict_button = st.button("🔮 Predict Loan Approval", use_container_width=True)

# Display mock prediction (for UI testing)
if predict_button:

    credit_history_input = 0.0
    if credit_history == "Yes (1.0)":
        credit_history_input = 1.0

    sample_data = [[
        loan_amount,                            # e.g.: 150.0
        loan_term,                              # e.g.: 360.0
        applicant_income + coapplicant_income,  # e.g.: 8000.0
        gender,                                 # e.g.: 'Male',
        married,                                # e.g.: 'Yes',
        dependents,                             # e.g.: '0',
        education,                              # e.g.: 'Graduate',
        self_employed,                          # e.g.: 'No',
        credit_history_input,                   # e.g.: 1.0
        property_area                           # e.g.: 'Semiurban'
    ]]

    predict_proba, prediction = predict_sample_data(model_data, sample_data, cutoff)

    st.toast("✅ Prediction complete!")

    if prediction:
        st.write("Eligible for loan!")
        st.metric('Predicted probability', np.round(predict_proba, 2),
                  f"{float(predict_proba - cutoff):.1%} above cutoff")
    else:
        st.write("Sorry, you're not elegible")
        st.metric('Predicted probability', np.round(predict_proba, 2),
                  f"{float(predict_proba - cutoff):.1%} below cutoff") 