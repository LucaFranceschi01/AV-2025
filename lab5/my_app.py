import streamlit as st

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
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income ($)", min_value=150, step=100, value=81000)
    coapplicant_income = st.number_input("Co-Applicant Income ($)", min_value=0, step=100, value=41700)

with col2:
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    credit_history = st.selectbox("Credit History Available?", ["Yes (1.0)", "No (0.0)"])
    loan_amount = st.number_input("Loan Amount ($k)", min_value=0, step=1, value=128)
    loan_term = st.number_input("Loan Term (Months)", min_value=1, step=1, value=360)

# Prediction Cutoff Section
st.subheader("Prediction Cutoff")
st.write("Set the probability cutoff for approval")
cutoff = st.slider("", 0.0, 1.0, 0.50, 0.01)

# Predict Button
st.write("")
predict_button = st.button("🔮 Predict Loan Approval", use_container_width=True)

# Display mock prediction (for UI testing)
if predict_button:
    st.success(f"Prediction complete! (Cutoff: {cutoff:.2f})")
    st.metric(label="Loan Approval Probability", value="0.78", delta="+28% above cutoff")
