import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

st.set_page_config(page_title='Data Exploration', layout='wide')

# ---------------------------------------------------------------------
# App UI
st.title('🔍 Data Exploration')
st.write('Dataset exploration and key insights')

# Load raw data
with st.spinner('Loading data...'):
    df = load_data()

st.header('Data Exploration')
st.markdown('Quick preview of raw data:')
st.dataframe(df.head())

st.subheader('Basic info')
st.write('Rows:', df.shape[0], 'Columns:', df.shape[1])

tmp = df.groupby(['car', 'model', 'body']).mean(numeric_only=True).sort_values('price', ascending=False).reset_index()
tmp['index'] = tmp.apply(lambda x: f"{x['car']}: {x['model']} ({x['body']})", axis=1)
fig, ax = plt.subplots(figsize=(8,6))
sns.barplot(data=tmp.head(10), x='price', y='index', ax=ax)
ax.set_title('Top 10 most expensive cars')
st.pyplot(fig)

columns = [x for x in df.columns if x not in ['car','model']]
n_cols = 3
n_rows = (len(columns) + 2) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
axes = axes.flatten()
for i, col in enumerate(columns):
    ax = axes[i]
    if df[col].dtype == 'object' or df[col].nunique() < 10:
        sns.countplot(x=col, data=df, ax=ax)
        ax.set_title(f'Count Plot of {col}')
        plt.setp(ax.get_xticklabels(), rotation=45)
    else:
        sns.kdeplot(data=pd.DataFrame(df[col]), ax=ax, fill=True)
        ax.set_title(f'Density Plot of {col}')
for j in range(i+1, n_rows*n_cols):
    fig.delaxes(axes[j])
fig.tight_layout()
st.pyplot(fig)

corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='icefire', ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)
