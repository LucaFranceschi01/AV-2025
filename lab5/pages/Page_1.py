# ------------------------------------------------------------
# Loan Data Analysis
# Author: Luca Franceschi
# Email: luca.franceschi01@estudiant.upf.edu
# All contents are created in-house unless explicitly stated.

# UI Created with assistance from ChatGPT (GPT-5, OpenAI)
# ------------------------------------------------------------

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data

data = load_data('loan_dataset.csv')

st.set_page_config(page_title="LoanTech | We show the data", layout="wide")

st.title("Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.subheader('Loan Status Distribution')
    st.write('Overall Loan Status (Y=Yes, N=No)')

    fig = px.pie(data_frame=data, names='Loan_Status', labels={'Y': 'Yes', 'N': 'No'}, hole=.3)

    st.plotly_chart(fig)

with col2:
    st.subheader('Credit History vs. Loan Status')
    st.write('Impact of Credit Score on Approval')

    data = data.sort_values('Loan_Status', ascending=True) # for visualization purposes

    fig = go.Figure()
    fig.add_trace(go.Histogram(histfunc="count", x=data[data['Loan_Status'] == 'Y']['Credit_History'], name='Y'))
    fig.add_trace(go.Histogram(histfunc="count", x=data[data['Loan_Status'] == 'N']['Credit_History'], name='N'))

    st.plotly_chart(fig)