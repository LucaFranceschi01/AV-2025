'''
This application's UI is largely programmed by gemini. The logic, data processing, inference, etc. is not.
'''

import streamlit as st

st.set_page_config(
    page_title="Bankruptcy Prediction",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Corporate Bankruptcy Prediction System")

st.markdown("""
### Welcome to the Bankrupcy Prediction Dashboard

This application analyzes financial data to predict potential bankruptcy. 

**Navigation:**
1. **Overview & Comparison:** EDA of the data and comparison of different ML models (RF, SVM, XGB, LogReg).
2. **Model Performance:** Deep dive into the chosen **Logistic Regression** model.
3. **Explainability (XAI):** Understanding *why* the model makes specific decisions using SHAP.

**Why Logistic Regression?**
Even though XGBoost has slightly clearer explainability, the improved accuracy and speed of Logistic Regression has won this battle.
""")

st.info("Select a page from the sidebar to begin.")