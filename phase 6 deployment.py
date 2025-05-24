import streamlit as st
import joblib
import json
import pandas as pd

# --- Load ML Components ---
scaler = joblib.load('scaler.pkl')
model = joblib.load('random_forest_model.pkl')
with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

categorical_cols = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                    'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']

label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = joblib.load(f'label_encoder_{col}.pkl')

# --- Page Config ---
st.set_page_config(page_title="🚗 Used Car Price Predictor", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("🚙 Car Price App")
    st.markdown("A smart tool to predict the price of a used car based on specifications.")
    st.image("https://img.freepik.com/free-vector/car-rental-abstract-concept-illustration_335657-3080.jpg", use_column_width=True)

# --- Main Title ---
st.markdown("<h1 style='text-align: center;'>Used Car Price Prediction 💰</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter the car specifications below and get a price prediction instantly!</p>", unsafe_allow_html=True)

# --- Model Info ---
with st.expander("📈 Model Information", expanded=False):
    st.markdown("""
    **Model:** Random Forest Regressor  
    **Performance Metrics:**  
    - 🔹 R² Score: **0.897**  
    - 🔹 MAE: **918.96**  
    - 🔹 RMSE: **1,318.78**
    """)

st.markdown("---")
st.subheader("🚘 Car Specifications")

# --- Input Layout ---
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    symboling = st.number_input("🔢 Symboling (Risk Factor)", min_value=-3, max_value=3, value=0)
    fueltype = st.selectbox("⛽ Fuel Type", options=label_encoders['fueltype'].classes_)
    aspiration = st.selectbox("🌀 Aspiration", options=label_encoders['aspiration'].classes_)
    doornumber = st.selectbox("🚪 Number of Doors", options=label_encoders['doornumber'].classes_)
    carbody = st.selectbox("🚗 Car Body Type", options=label_encoders['carbody'].classes_)
    drivewheel = st.selectbox("🛞 Drive Wheel Type", options=label_encoders['drivewheel'].classes_)

with col2:
    enginelocation = st.selectbox("🧩 Engine Location", options=label_encoders['enginelocation'].classes_)
    enginetype = st.selectbox("🛠️ Engine Type", options=label_encoders['enginetype'].classes_)
    cylindernumber = st.selectbox("🔘 Cylinder Count", options=label_encoders['cylindernumber'].classes_)
    fuelsystem = st.selectbox("🧯 Fuel System", options=label_encoders['fuelsystem'].classes_)
    enginesize = st.number_input("📏 Engine Size (cc)", value=130)
    horsepower = st.number_input("⚡ Horsepower", value=111)

with col3:
    wheelbase = st.number_input("📐 Wheelbase (in)", value=88.6)
    carlength = st.number_input("📏 Car Length (in)", value=168.8)
    carwidth = st.number_input("📏 Car Width (in)", value=64.1)
    carheight = st.number_input("📏 Car Height (in)", value=48.8)
    curbweight = st.number_input("⚖️ Curb Weight (lbs)", value=2548)

st.markdown("---")
st.subheader("🔧 Engine & Performance Details")

col4, col5, col6 = st.columns(3)

with col4:
    boreratio = st.number_input("🛞 Bore Ratio", value=3.47)
    stroke = st.number_input("🛞 Stroke", value=2.68)

with col5:
    compressionratio = st.number_input("📉 Compression Ratio", value=9.0)
    peakrpm = st.number_input("🔺 Peak RPM", value=5000)

with col6:
    citympg = st.number_input("🏙️ City MPG", value=21)
    highwaympg = st.number_input("🛣️ Highway MPG", value=27)

# --- Prediction Logic ---
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

if st.button("🎯 Predict Car Price"):
    input_df = pd.DataFrame([input_data_dict])

    # Encode categorical variables
    for col in categorical_cols:
        le = label_encoders[col]
        input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Ensure all feature columns exist
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]

    # Scale and Predict
    input_scaled = scaler.transform(input_df)
    predicted_price = model.predict(input_scaled)

    # --- Output ---
    st.success(f"💸 **Estimated Car Price:** ${predicted_price[0]:,.2f}")
    st.balloons()
