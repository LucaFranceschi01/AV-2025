import streamlit as st
from PIL import Image

# App name and styling
st.set_page_config(page_title="AutoValuator", layout="wide")

# Header
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color:#2E86C1;'>AutoValuator</h1>
        <h3>Your smart companion for car price recommendations</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔍 Data Exploration")
    st.markdown("Explore the dataset, understand distributions, correlations, and gain insights.")
    if st.button("Go to Data Exploration"):
        st.switch_page("pages/1_Data_Exploration.py")

with col2:
    st.markdown("### 🚗 Price Prediction")
    st.markdown("Get a suggested selling price based on your vehicle's characteristics.")
    if st.button("Go to Prediction"):
        st.switch_page("pages/2_Prediction_Playground.py")

with col3:
    st.markdown("### 📊 Model Explainability")
    st.markdown("Understand how the model makes decisions using interpretability tools.")
    if st.button("Go to Explainability"):
        st.switch_page("pages/3_Model_Explainability.py")
