import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score
import pickle

st.set_page_config(page_title="Overview & Comparison", page_icon="📊", layout="wide")

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

st.write(df_sample.head())

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
    target_corr = corr['Bankrupt?'].sort_values(ascending=False).head(10).drop('Bankrupt?')
    fig_corr = px.bar(x=target_corr.values, y=target_corr.index, orientation='h',
                      title="Top Features Correlated with Bankruptcy",
                      labels={'x': 'Correlation', 'y': 'Feature'})
    st.plotly_chart(fig_corr, use_container_width=True)

st.subheader("Full Feature Correlation Heatmap")
st.markdown("""
A visual representation of the correlation between the top 15 most correlated features.
""")

N_FEATURES = 15
top_features = corr['Bankrupt?'].abs().sort_values(ascending=False).head(N_FEATURES + 1).index.tolist()

corr_top = df_sample[top_features].corr(numeric_only=True).round(2)

fig_heatmap = px.imshow(
    corr_top,
    text_auto=True,  # Annotate with the correlation value
    aspect="auto",
    color_continuous_scale=px.colors.diverging.RdBu,
    color_continuous_midpoint=0,
    title="Correlation Heatmap of Top 15 Features"
)

fig_heatmap.update_xaxes(side="bottom")
fig_heatmap.update_layout(
    height=800, # Set a fixed height for better visibility
    margin=dict(l=50, r=50, b=100, t=100) # Increase margins for labels
)

st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()
st.header("2. Predictive Feature Distributions")
st.write("Comparing the distributions of the most relevant features between the two classes.")

# Features selected based on correlation and financial significance
relevant_features = [
    'Net Income to Total Assets',
    'Operating Gross Margin',
    'Total debt/Total net worth',
    'Current Ratio'
]

col1, col2 = st.columns(2)
df_sample_cp = df_sample.copy()

df_sample_cp['Total debt/Total net worth'] = np.log1p(df_sample_cp['Total debt/Total net worth'])
df_sample_cp['Current Ratio'] = np.log1p(df_sample_cp['Current Ratio'])

for i, feature in enumerate(relevant_features):

    # --- HISTOGRAM WITH DENSITY OVERLAY ---
    fig_hist = px.histogram(
        df_sample_cp,
        x=feature,
        color='Bankrupt?',
        marginal="box", # Adds a box plot on top for outlier context
        histnorm='probability density', # Normalizes bars so the area is 1 for each class
        opacity=0.6,
        barmode='overlay',
        title=f'Density Distribution of {feature}'
    )

    # Customize layout
    fig_hist.update_layout(
        yaxis_title="Probability Density",
        legend_title="Bankrupt?"
    )

    # Place plots in columns
    if i % 2 == 0:
        with col1:
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        with col2:
            st.plotly_chart(fig_hist, use_container_width=True)

# --- MODEL COMPARISON SECTION ---
st.divider()
st.header("3. Model Comparison (ROC Curves)")

st.markdown("""
We evaluated Random Forest, SVM, Logistic Regression, XGBoost, and simplified XGBoost with the top 20 features.
The plot below compares their **True Positive Rate** vs. **False Positive Rate**.
""")

fig_roc = go.Figure()
fig_roc.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

for model_name, y_proba in probas.items():

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
    width=900, height=800
)

left, middle, right = st.columns((1, 5, 1))
with middle:
    st.plotly_chart(fig_roc, use_container_width=True)


st.success("**Decision:** Logistic Regression chosen for deployment due to high AUC (comparable to XGBoost) and similar interpretability.")