import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder


st.set_page_config(page_title='Data Exploration', layout='wide')

st.title('🔍 Data Exploration')
st.write('Dataset exploration and key insights')

with st.spinner('Loading data...'):
    df = load_data()

st.header('Data Exploration')
st.markdown('Quick preview of raw data:')
st.dataframe(df.head())

st.subheader('Basic info')
st.write('Rows:', df.shape[0], 'Columns:', df.shape[1])

col1, col2 = st.columns(2)

with col1:
    with st.spinner('Loading chart...'):
        tmp = df.groupby(['car', 'model', 'body']).mean(numeric_only=True).sort_values('price', ascending=False).reset_index()
        tmp['index'] = tmp.apply(lambda x: f"{x['car']}: {x['model']} ({x['body']})", axis=1)
        fig = px.bar(tmp[:10], x='price', y='index', orientation='h', title='Top 10 Most Expensive Cars',
                    color_continuous_scale=my_colorscale, color='price')
        fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    with st.spinner('Loading chart...'):
        tmp = df.groupby(['car']).mean(numeric_only=True).sort_values('price', ascending=False).reset_index()
        fig = px.bar(tmp[:10], x='price', y='car', orientation='h', title='Top 10 most expensive brands on average',
                    color_continuous_scale=my_colorscale, color='price')
        fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
        st.plotly_chart(fig, use_container_width=True)

df = df[df["price"] <= 100000]
df = df[df["price"] >= 1000]
df = df[df["mileage"] <= 600]
df = df[df["engV"] <= 7.5]
df = df[df["year"] >= 1975]

# Select columns except car and model
columns = [x for x in df.columns if x not in ['car', 'model']]

n_cols = 3
n_rows = (len(columns) + n_cols - 1) // n_cols

with st.spinner('Loading chart...'):
    # Create subplot grid
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f"{col}" for col in columns])

    # Loop through columns
    for idx, col in enumerate(columns):
        r = idx // n_cols + 1
        c = idx % n_cols + 1

        # Categorical or low-cardinality count plot
        if df[col].dtype == 'object' or df[col].nunique() < 10:
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, 'count']

            fig.add_trace(
                go.Bar(
                    x=counts[col],
                    y=counts['count'],
                    width=0.9
                ),
                row=r, col=c
            )

        # Numeric density plot
        else:
            fig.add_trace(
                go.Histogram(
                    x=df[col],
                    histnorm='probability density',
                ),
                row=r, col=c
            )

    # Update layout
    fig.update_layout(
        height=300 * n_rows,
        width=1200,
        showlegend=False,
        title="Feature Distributions",
    )

    st.plotly_chart(fig, use_container_width=True)


with st.spinner('Loading chart...'):
    target = "price"
    features = [x for x in df.columns if x not in ["car", "model", target]]

    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols

    # Create subplot grid
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"{feature} vs {target}" for feature in features]
    )

    # Loop through features
    for idx, feature in enumerate(features):
        r = idx // n_cols + 1
        c = idx % n_cols + 1

        # Categorical / low-cardinality -> boxplot
        if df[feature].dtype == "object" or df[feature].nunique() < 10:
            fig.add_trace(
                go.Box(
                    x=df[feature],
                    y=df[target],
                    boxmean=True,
                ),
                row=r, col=c
            )

        # Numeric -> scatter plot
        else:
            fig.add_trace(
                go.Scatter(
                    x=df[feature],
                    y=df[target],
                    mode='markers',
                    opacity=0.6,
                    marker=dict(
                        colorscale=my_colorscale,
                        color=df[target],
                        showscale=True,
                        cmin=df[target].min(),
                        cmax=df[target].max(),
                    )
                ),
                row=r, col=c
            )

    # Layout options
    fig.update_layout(
        height=350 * n_rows,
        width=1200,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

le_car = LabelEncoder()
df['car'] = le_car.fit_transform(df['car'])

le_body = LabelEncoder()
df['body'] = le_body.fit_transform(df['body'])

le_engType = LabelEncoder()
df['engType'] = le_engType.fit_transform(df['engType'])

le_drive = LabelEncoder()
df['drive'] = le_drive.fit_transform(df['drive'])

corr = df.corr(numeric_only=True)

with st.spinner('Loading chart...'):
    fig = px.imshow(
        corr,
        text_auto=True,
        zmin=-1,
        zmax=1,
        color_continuous_scale=my_colorscale,
        title="Correlation Heatmap"
    )

    fig.update_layout(
        height=800,
        width=900
    )

    st.plotly_chart(fig, use_container_width=True)
