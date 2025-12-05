import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_logreg_model():
    with open('assets/logreg_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_test_data():
    X_test = pd.read_csv('assets/X_test.csv')
    y_test = np.load('assets/y_test.npy')
    return X_test, y_test

model = load_logreg_model()
X_test, y_test = load_test_data()

st.title("Logistic Regression Performance")

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Confusion Matrix (Normalized by Predicted Class)")
    
    # 1. Calculate raw confusion matrix
    cm_raw = confusion_matrix(y_test, y_pred)
    
    # 2. Normalize by Predicted Class (normalize='pred')
    # Sum of each column is 1.0 (True Positives / All Predicted Positives, etc.)
    
    # The sum of each column (Predicted classes)
    col_sums = cm_raw.sum(axis=0, keepdims=True)

    # Avoid division by zero
    col_sums[col_sums == 0] = 1 
    
    # Normalized matrix
    cm_normalized = cm_raw / col_sums
    
    # --- PLOTLY SETUP ---
    x = ['Predicted 0', 'Predicted 1']
    y = ['Actual 1', 'Actual 0'] # Reversed for standard layout match

    # z should hold the normalized values (0 to 1)
    # z[0][0] = CM[1][0] (False Negatives - Normalized)
    # z[0][1] = CM[1][1] (True Positives - Normalized)
    # z[1][0] = CM[0][0] (True Negatives - Normalized)
    # z[1][1] = CM[0][1] (False Positives - Normalized)
    
    # Note: We reverse the row order for Plotly's standard heatmap convention (y=Actual 1 top)
    z = [[cm_normalized[1][0], cm_normalized[1][1]], 
         [cm_normalized[0][0], cm_normalized[0][1]]]
    
    # Annotation text should show the normalized value formatted as percentage
    z_text = [[f'{v:.2f} ({cm_raw[r][c]})' for c, v in enumerate(row)] 
              for r, row in enumerate(z)]
    
    fig_cm = ff.create_annotated_heatmap(
        z, x=x, y=y, annotation_text=z_text, colorscale='Blues'
    )

    fig_cm.update_layout(title_text='<i>Normalized by Predicted Class (Precision)</i>')
    
    st.plotly_chart(fig_cm, use_container_width=True)

with col2:
    st.subheader("Probability Distribution")
    df_prob = pd.DataFrame({'Probability': y_proba, 'Actual': y_test})
    df_prob['Actual'] = df_prob['Actual'].map({False: 'Non-Bankrupt', True: 'Bankrupt'})

    
    fig_hist = px.histogram(
        df_prob, x="Probability", color="Actual", nbins=50,
        title="Distribution of Predicted Probabilities",
        opacity=0.7, barmode='overlay'
    )
    fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
    st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Classification Metrics")
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.dataframe(df_report.style.format("{:.3f}"))