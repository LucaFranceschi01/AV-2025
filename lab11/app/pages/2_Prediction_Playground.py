import streamlit as st
import numpy as np
from utils import *

st.set_page_config(page_title='Price Prediction', layout='wide')

st.title('🚗 Price Prediction')
st.write('Provide attributes for a sample car to get a predicted price.')

model_loaded = load_model()
df = load_data()

col_sliders, col_selects = st.columns(2)

with col_selects:
    year_in = st.slider('Year', min_value=1950, max_value=2025, value=2010)
    mileage_in = st.slider('Mileage (thousands?)', min_value=0, max_value=1000, value=100, step=10)
    engV_in = st.slider('engV', min_value=0.0, max_value=30.0, value=1.8, step=0.5)

with col_sliders:
    car_in = st.selectbox('Brand (car)', np.sort(df['car'].unique()))
    body_in = st.selectbox('Body', np.sort(df['body'].unique()))
    engType_in = st.selectbox('Engine Type', np.sort(df['engType'].unique()))
    registration_in = st.selectbox('Registered?', ['yes','no'])
    drive_in = st.selectbox('Drive', np.sort(df['drive'].unique()[:3]))

if st.button('Predict'):
    if model_loaded is None:
        st.warning('Please load or train a model first.')
    else:
        X_sample = np.array([[car_in, body_in, mileage_in, engV_in, engType_in, registration_in, year_in, drive_in]], dtype=object)

        st.write(predict_sample_data(model_loaded, X_sample))
