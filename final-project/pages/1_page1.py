import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score
import pickle

st.set_page_config(page_title="Overview & Comparison", page_icon="📊", layout="wide")

# --- LOAD ASSETS ---
@st.cache_data
def load_comparison_data():
    y_test = np.load('assets/y_test.npy')
    df_sample = pd.read_csv('assets/df_sample.csv')
    with open('assets/model_comparison_probas.pkl', 'rb') as f:
        probas = pickle.load(f)
    return y_test, df_sample, probas

y_test, df_sample, probas = load_comparison_data()

# --- EDA SECTION ---
st.header("1. Exploratory Data Analysis (Sample)")
st.write("A high-level look at the data distribution and correlations.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Target Distribution")
    fig_hist = px.histogram(df_sample, x="Bankrupt?", color="Bankrupt?", 
                            title="Bankruptcy Distribution (Imbalanced Data)")
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.subheader("Correlation Matrix (Top Features)")
    # Calculate corr for sample to be fast
    corr = df_sample.corr(numeric_only=True)
    # Select correlation with target
    target_corr = corr['Bankrupt?'].sort_values(ascending=False).head(10)
    fig_corr = px.bar(x=target_corr.values, y=target_corr.index, orientation='h',
                      title="Top Features Correlated with Bankruptcy",
                      labels={'x': 'Correlation', 'y': 'Feature'})
    st.plotly_chart(fig_corr, use_container_width=True)

# --- MODEL COMPARISON SECTION ---
st.divider()
st.header("2. Model Comparison (ROC Curves)")

st.markdown("""
We evaluated Random Forest, SVM, Logistic Regression, and XGBoost. 
The plot below compares their **True Positive Rate** vs. **False Positive Rate**.
""")

fig_roc = go.Figure()
fig_roc.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

for model_name, y_proba in probas.items():
    # Handle simplified xgboost length mismatch if any, usually handled in generation
    # Assuming lengths match y_test
    if len(y_proba) == len(y_test):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', 
            name=f'{model_name} (AUC={auc:.3f})'
        ))

fig_roc.update_layout(
    title="ROC Curve Comparison",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    width=900, height=600
)

st.plotly_chart(fig_roc, use_container_width=True)

st.success("✅ **Decision:** Logistic Regression chosen for deployment due to high AUC (comparable to XGBoost) and superior interpretability.")