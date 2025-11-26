import streamlit as st
import matplotlib.pyplot as plt

from utils import *

st.set_page_config(page_title='Model Explainability', layout='wide')

st.title('📊 Model Explainability')
st.write('Global and local explainability using SHAP')

# Load model and encoders
try:
    data = load_model('models/model.pkl')
    if data:
        model = data['model']
        le_car = data['le_car']
        le_body = data['le_body']
        le_engType = data['le_engType']
        le_drive = data['le_drive']
    else:
        st.write('The model was not loaded correctly!!')

    st.success('Loaded saved model and encoders')
except Exception as e:
    st.error('Could not load model.pkl from models/. You can train a model on the Model page first. Error: '+ str(e))
    model = None

df = load_data_clean()
X = df.drop('price', axis=1)

if model is not None:
    if st.button('Compute SHAP values (this may take a moment)'):
        with st.spinner('Computing explainer...'):
            explainer, shap_values = get_shap_explainer(model, X)
            st.success('Explainer computed')
            st.subheader('Global explainability')
            st.markdown('### Mean feature importance (bar)')
            # shap.plots.bar uses matplotlib; capture to figure
            fig = plt.figure(figsize=(8,6))
            shap.plots.bar(shap_values)
            st.pyplot(fig)

            st.markdown('### Summary plot (beeswarm)')
            fig2 = plt.figure(figsize=(10,6))
            shap.summary_plot(shap_values, X)
            st.pyplot(fig2)

            st.markdown('### Dependence plots for top features')
            top_feats = list(pd.DataFrame(shap_values.values, columns=X.columns).abs().mean().sort_values(ascending=False).head(3).index)
            for f in top_feats:
                st.write(f'Dependence plot for {f}')
                figd = plt.figure(figsize=(8,4))
                try:
                    shap.dependence_plot(f, shap_values.values, X, interaction_index=None, show=False)
                except Exception:
                    shap.dependence_plot(f, shap_values, X, interaction_index=None, show=False)
                st.pyplot(figd)

            st.markdown('---')
            st.subheader('Local explainability')
            idx = st.number_input('Choose an index from X (row) to explain locally', min_value=0, max_value=len(X)-1, value=0)
            sv = shap_values[idx]
            st.write('Base value (average prediction):', sv.base_values)
            st.write('Model prediction for row:', model.predict(X.iloc[[idx]])[0])

            st.write('Waterfall plot (local)')
            figw = plt.figure(figsize=(8,6))
            shap.plots.waterfall(sv)
            st.pyplot(figw)

            st.write('Decision plot (local)')
            figdec = plt.figure(figsize=(10,4))
            shap.decision_plot(sv.base_values, sv.values, X.iloc[idx], show=False)
            st.pyplot(figdec)

            st.success('SHAP visualizations rendered')

    st.info('To compute explanations, click the button above. For a faster workflow train a model first on the Model & Prediction page and save under models/model.pkl')
