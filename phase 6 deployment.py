import streamlit as st
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Load scaler, model, and feature columns
scaler = joblib.load('scaler.pkl')
model = joblib.load('random_forest_model.pkl')

with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

categorical_cols = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                    'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']

label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = joblib.load(f'label_encoder_{col}.pkl')

# --- Streamlit UI ---
st.set_page_config(page_title="Used Car Price Predictor", layout="wide")

st.title("🚗 Used Car Price Prediction")

st.markdown("""
Welcome! Fill in the car details below and click **Predict Price** to estimate the car's value.
""")

# Display model info
with st.expander("Model Information ℹ️", expanded=True):
    st.markdown("""
    **Model Type:** Random Forest Regressor  
    **Train R² Score:** 0.977  
    **Test MAE:** 918.96  
    **Test MSE:** 1,739,172.29  
    **Test RMSE:** 1,318.78  
    **Test R² Score:** 0.897  
    """)

# Input sections
st.header("Car Specifications")

col1, col2, col3 = st.columns(3)

with col1:
    symboling = st.number_input("Symboling (Risk factor)", min_value=-3, max_value=3, value=3)
    fueltype = st.selectbox("Fuel Type", options=list(label_encoders['fueltype'].classes_))
    aspiration = st.selectbox("Aspiration", options=list(label_encoders['aspiration'].classes_))
    doornumber = st.selectbox("Door Number", options=list(label_encoders['doornumber'].classes_))
    carbody = st.selectbox("Car Body", options=list(label_encoders['carbody'].classes_))
    drivewheel = st.selectbox("Drive Wheel", options=list(label_encoders['drivewheel'].classes_))

with col2:
    enginelocation = st.selectbox("Engine Location", options=list(label_encoders['enginelocation'].classes_))
    enginetype = st.selectbox("Engine Type", options=list(label_encoders['enginetype'].classes_))
    cylindernumber = st.selectbox("Cylinder Number", options=list(label_encoders['cylindernumber'].classes_))
    fuelsystem = st.selectbox("Fuel System", options=list(label_encoders['fuelsystem'].classes_))
    enginesize = st.number_input("Engine Size (cc)", value=130)
    horsepower = st.number_input("Horsepower", value=111)

with col3:
    wheelbase = st.number_input("Wheelbase (inches)", value=88.6)
    carlength = st.number_input("Car Length (inches)", value=168.8)
    carwidth = st.number_input("Car Width (inches)", value=64.1)
    carheight = st.number_input("Car Height (inches)", value=48.8)
    curbweight = st.number_input("Curb Weight (lbs)", value=2548)

st.header("Engine Details")

col4, col5, col6 = st.columns(3)

with col4:
    boreratio = st.number_input("Bore Ratio", value=3.47)
    stroke = st.number_input("Stroke", value=2.68)

with col5:
    compressionratio = st.number_input("Compression Ratio", value=9.0)
    peakrpm = st.number_input("Peak RPM", value=5000)

with col6:
    citympg = st.number_input("City MPG", value=21)
    highwaympg = st.number_input("Highway MPG", value=27)

input_data_dict = {
    'symboling': symboling,
    'fueltype': fueltype,
    'aspiration': aspiration,
    'doornumber': doornumber,
    'carbody': carbody,
    'drivewheel': drivewheel,
    'enginelocation': enginelocation,
    'wheelbase': wheelbase,
    'carlength': carlength,
    'carwidth': carwidth,
    'carheight': carheight,
    'curbweight': curbweight,
    'enginetype': enginetype,
    'cylindernumber': cylindernumber,
    'enginesize': enginesize,
    'fuelsystem': fuelsystem,
    'boreratio': boreratio,
    'stroke': stroke,
    'compressionratio': compressionratio,
    'horsepower': horsepower,
    'peakrpm': peakrpm,
    'citympg': citympg,
    'highwaympg': highwaympg
}

if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data_dict])

    # Encode categorical columns
    for col in categorical_cols:
        if col in input_df.columns:
            le = label_encoders[col]
            input_df[col] = input_df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # Add missing columns if any
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]
    input_scaled = scaler.transform(input_df)
    predicted_price = model.predict(input_scaled)

    st.success(f"### Predicted Car Price: ${predicted_price[0]:,.2f}")

    # --- Feature Importance Plot ---
    st.header("🔍 Model Explainability")
    with st.expander("📊 Feature Importance (Random Forest)"):
        importances = model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_imp_df.head(20), x='Importance', y='Feature', palette="Blues_d", ax=ax)
        ax.set_title("Top 20 Most Important Features")
        st.pyplot(fig)

    # --- SHAP Plot ---
    with st.expander("📈 SHAP Values (Model Interpretability)"):
        st.markdown("SHAP helps explain how each feature impacts the prediction.")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)
        shap.initjs()

        fig_shap = shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0][0], input_df.iloc[0])
        st.pyplot(fig_shap, clear_figure=True)
