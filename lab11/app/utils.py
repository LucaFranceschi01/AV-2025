import streamlit as st
import pandas as pd
import pickle
import shap
import numpy as np
import time

# from https://plotly.com/python/colorscales/
my_colorscale = [[0.0, "rgb(49,54,149)"],
        [0.1111111111111111, "rgb(69,117,180)"],
        [0.2222222222222222, "rgb(116,173,209)"],
        [0.3333333333333333, "rgb(171,217,233)"],
        [0.4444444444444444, "rgb(224,243,248)"],
        [0.5555555555555556, "rgb(254,224,144)"],
        [0.6666666666666666, "rgb(253,174,97)"],
        [0.7777777777777778, "rgb(244,109,67)"],
        [0.8888888888888888, "rgb(215,48,39)"],
        [1.0, "rgb(165,0,38)"]]

@st.cache_resource
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

@st.cache_data
def load_data(path='car_ad_display.csv'):
    time.sleep(2)
    
    df = pd.read_csv(path, encoding='ISO-8859-1', sep=';')

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')

    car_map = shorten_categories(df.car.value_counts(), 10)
    df['car'] = df['car'].map(car_map)

    model_map = shorten_categories(df.model.value_counts(), 10)
    df['model'] = df['model'].map(model_map)

    return df

@st.cache_data
def load_data_clean(path='car_ad_display_clean.csv'):
    df = pd.read_csv(path, encoding='ISO-8859-1', sep=',')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    return df

@st.cache_data
def load_X_test(path='X_test.csv'):
    df = pd.read_csv(path, encoding='ISO-8859-1', sep=',')
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
def get_shap_explainer(_model, X_sample):
    # Using TreeExplainer if possible for speed; fallback to general Explainer
    try:
        explainer = shap.Explainer(_model)
    except Exception:
        explainer = shap.Explainer(_model)
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