import streamlit as st
import pandas as pd
import pickle
import shap
import numpy as np

@st.cache_data
def load_data(path='car_ad_display.csv'):
    df = pd.read_csv(path, encoding='ISO-8859-1', sep=';')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    return df

@st.cache_data
def load_data_clean(path='car_ad_display_clean.csv'):
    df = pd.read_csv(path, encoding='ISO-8859-1', sep=';')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    return df

@st.cache_data
def load_model(path='models/model.pkl'):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file not found at: {path}")
        data = None
    except pickle.UnpicklingError:
        st.error("Error loading model: corrupt or incompatible pickle file.")
        data = None
    except Exception as e:
        st.error(f"Unexpected error while loading model: {e}")
        data = None
    return data

@st.cache_resource
def get_shap_explainer(model, X_sample):
    # Using TreeExplainer if possible for speed; fallback to general Explainer
    try:
        explainer = shap.Explainer(model)
    except Exception:
        explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    return explainer, shap_values

@st.cache_data
def predict_sample_data(_data, X_sample):
    
    model = _data["model"]
    le_car = _data["le_car"]
    le_body = _data["le_body"]
    le_engType = _data["le_engType"]
    le_drive = _data["le_drive"]

    df_original = load_data()
    yes_l = ['yes', 'YES', 'Yes', 'y', 'Y']

    # Apply the encoder and data type corrections:
    X_sample[:, 0] = str(X_sample[:, 0][0] if X_sample[:, 0][0] in list(df_original['car'].unique()) else 'Other')
    X_sample[:, 0] = le_car.transform(X_sample[:,0])
    X_sample[:, 1] = le_body.transform(X_sample[:,1])
    X_sample[:, 4] = le_engType.transform(X_sample[:,4])
    X_sample[:, 5] = int(1 if X_sample[:, 5][0] in yes_l else 0)
    X_sample[:, 7] = le_drive.transform(X_sample[:,7])

    X_sample = np.array([[
        int(X_sample[0, 0]),
        int(X_sample[0, 1]),
        int(X_sample[0, 2]),
        float(X_sample[0, 3]),
        int(X_sample[0, 4]),
        int(X_sample[0, 5]),
        int(X_sample[0, 6]),
        int(X_sample[0, 7])
    ]])

    y_pred_sample = model.predict(X_sample)

    return y_pred_sample[0]