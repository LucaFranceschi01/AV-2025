'''
This application's UI is largely programmed by gemini. The logic, data processing, inference, etc. is not.
'''

import streamlit as st

st.set_page_config(
    page_title="Bankruptcy Prediction AI",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Corporate Bankruptcy Prediction System")

st.markdown("""
### Welcome to the AI Model Dashboard

This application analyzes financial data to predict potential bankruptcy. 

**Navigation:**
1. **Overview & Comparison:** EDA of the data and comparison of different ML models (RF, SVM, XGB, LogReg).
2. **Model Performance:** Deep dive into the chosen **Logistic Regression** model.
3. **Explainability (XAI):** Understanding *why* the model makes specific decisions using SHAP.

**Why Logistic Regression?**
While XGBoost offered marginally higher accuracy, we selected **Logistic Regression** for production because:
* **Interpretability:** It provides linear coefficients that are easy for financial auditors to validate.
* **Calibration:** It offers well-calibrated probabilities.
* **Efficiency:** Extremely fast inference time.
""")

st.info("Select a page from the sidebar to begin.")