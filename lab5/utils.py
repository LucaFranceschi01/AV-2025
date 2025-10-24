import pickle
import pandas as pd
import streamlit as st


@st.cache_resource # should work better than cache_data since the stored daya might be somewhat complex
def load_model(model_path: str):
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
    return data

@st.cache_data
def predict_sample_data(_data, sample_data, cutoff):

    raw_columns = ['LoanAmount', 'Loan_Amount_Term', 'TotalIncome', 
                'Gender', 'Married', 'Dependents', 
                'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    
    model = _data["model"]
    encoder = _data["encoder"]
    scaler = _data["scaler"]
    numeric_cols = _data["numeric_cols"]
    categorical_cols = _data["categorical_cols"]
    model_columns_ordered = _data["model_columns_ordered"]

    sample_df = pd.DataFrame(sample_data, columns=raw_columns)

    sample_df_scaled_data = scaler.transform(sample_df[numeric_cols])

    df_sample_scaled = pd.DataFrame(sample_df_scaled_data, 
                                columns=numeric_cols, 
                                index=sample_df.index) # Mantenim l'índex per al concat

    # Encode categorical cols
    sample_df_encoded_data = encoder.transform(sample_df[categorical_cols])

    encoded_col_names = encoder.get_feature_names_out(categorical_cols)
    df_sample_encoded = pd.DataFrame(sample_df_encoded_data, 
                                columns=encoded_col_names, 
                                index=sample_df.index) # Mantenim l'índex per al concat

    # Concat both dataframes horizontally
    sample_df_processed_df = pd.concat([df_sample_scaled, df_sample_encoded], axis=1)

    # Reorder cols (we need the same order as in training)
    sample_df_processed_df = sample_df_processed_df[model_columns_ordered]

    predict_proba = model.predict_proba(sample_df_processed_df)[:, 1]

    return predict_proba, (predict_proba >= cutoff).astype(int)