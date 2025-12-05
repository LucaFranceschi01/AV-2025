import streamlit as st
import shap
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Explainability", page_icon="🤖", layout="wide")

@st.cache_resource
def load_resources():
    with open('assets/logreg_model.pkl', 'rb') as f:
        model = pickle.load(f)
    X_test = pd.read_csv('assets/X_test.csv')
    return model, X_test

model, X_test = load_resources()

@st.cache_data
def get_shap_values(_model, _X_test):
    # use LinearExplainer for LogReg
    explainer = shap.LinearExplainer(_model, _X_test, model_output='raw')
    shap_values = explainer.shap_values(_X_test)
    
    # Check if list (binary class) or array
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    return explainer, shap_values

with st.spinner("Calculating SHAP values (this may take a moment)..."):
    explainer, shap_values = get_shap_values(model, X_test)

st.title("Explainable AI (XAI) Dashboard")
st.markdown("We use **SHAP (SHapley Additive exPlanations)** to understand feature contributions.")

# --- TABBED VIEW ---
tab1, tab2, tab3 = st.tabs(["Global Importance", "SHAP Values and eq. probabilities", "Local Prediction (Waterfall)"])

shap_values = np.array(shap_values)

# --- TAB 1: Global Feature Importance (Bar Plot) ---
with tab1:
    st.subheader("Which features matter most overall?")
    
    # Calculate mean absolute shap values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_names = X_test.columns
    
    df_import = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs_shap
    }).sort_values(by='Mean |SHAP|', ascending=True).tail(20)
    
    fig_bar = px.bar(df_import, x='Mean |SHAP|', y='Feature', orientation='h',
                     title="Top 20 Features by Mean Absolute SHAP Value", height=600)
    st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 2: Logit vs Probability ---
with tab2:
    st.subheader("Relationship between SHAP Logits and Probability")
    
    baseline_logit = explainer.expected_value
    
    # Calculate totals
    total_logits = np.sum(shap_values, axis=1) + baseline_logit
    probs = 1 / (1 + np.exp(-total_logits))
    
    df_scurve = pd.DataFrame({
        'Total Logit': total_logits,
        'Probability': probs
    })
    
    fig_curve = px.scatter(df_scurve, x='Total Logit', y='Probability', 
                           title="Sigmoid Curve: Total Logit vs Probability",
                           opacity=0.5)
    
    fig_curve.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
    fig_curve.update_layout(height=600)
    
    st.plotly_chart(fig_curve, use_container_width=True)

# --- TAB 3: Local Explanation (Waterfall) ---
with tab3:
    st.subheader("Why did the model predict Bankruptcy for a specific company?")
    
    # Filter for high risk companies for interest
    preds = model.predict(X_test)
    risky_indices = np.where(preds == 1)[0]
    
    if len(risky_indices) > 0:
        selected_idx = st.selectbox("Select a High-Risk Test Sample Index:", risky_indices, accept_new_options=True)
    else:
        selected_idx = st.number_input("Select Index", min_value=0, max_value=len(X_test)-1, value=0)
    selected_idx = int(selected_idx)
    # Get values for this instance
    row_shap = shap_values[selected_idx]
    row_features = X_test.iloc[selected_idx]
    base_value = explainer.expected_value
    
    # Create Waterfall Data
    # Sort by magnitude for cleaner plot
    sort_inds = np.argsort(np.abs(row_shap))
    top_n = 15
    top_inds = sort_inds[-top_n:]
    
    # Calculate 'Other' group
    other_shap = np.sum(row_shap[sort_inds[:-top_n]])
    
    plot_names = list(row_features.index[top_inds]) + ["Rest of features"]
    plot_values = list(row_shap[top_inds]) + [other_shap]
    plot_text = [f"{val:.2f}" for val in row_features.values[top_inds]] + [""]
    
    # Determine measures
    measures = ["relative"] * (len(plot_values))
    # We construct the waterfall relative to base
    
    fig_waterfall = go.Figure(go.Waterfall(
        name = "20", orientation = "h",
        measure = measures,
        y = plot_names,
        x = plot_values,
        text = plot_text,
        base = base_value,
        decreasing = {"marker":{"color":"#FF4136"}}, # Red for bankrupt
        increasing = {"marker":{"color":"#2ECC40"}}, # Green for safe
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    
    final_logit = np.sum(row_shap) + base_value
    final_prob = 1 / (1 + np.exp(-final_logit))
    
    fig_waterfall.update_layout(
        title = f"Waterfall Chart (Logit Scale) | Final Prob: {final_prob:.4f}",
        xaxis_title = "Contribution to Log-Odds (Logit)",
        height = 700
    )
    
    st.plotly_chart(fig_waterfall, use_container_width=True)
    
    st.info(f"Base Value (Average Logit): {base_value:.4f} | Final Logit: {final_logit:.4f}")