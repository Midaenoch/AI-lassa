import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ----------------------------
# Load ML components
# ----------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scallernew.pkl")
    label_encoder = joblib.load("label_encodersnew.pkl")
    model = joblib.load("svmnew_model.pkl")
    return scaler, label_encoder, model

scaler, label_encoder, model = load_artifacts()

st.set_page_config(page_title="Patient Outcome Predictor", layout="wide")
st.title("🏥 Patient Outcome Prediction App")

st.write("This app predicts **Outcome (Positive/Negative)** using a trained SVM model.")

# =====================================================
# Feature list to keep order consistent
# =====================================================
expected_features = [
    "Age", "Sex", "WardType",
    "Temperature_C", "HeartRate_bpm", "RespRate_bpm",
    "SystolicBP_mmHg", "DiastolicBP_mmHg", "SpO2_%",
    "GCS", "OxygenNeeded", "VentilationNeeded", "BleedingPresent"
]

# ==============================
# Manual Entry
# ==============================
with st.expander("✍️ Manual Entry"):
    with st.form("manual_form"):
        # Demographics
        age = st.number_input("Age (years)", min_value=0, max_value=120, step=1)
        sex = st.selectbox("Sex", ["M", "F"])
        ward_icu = st.selectbox("Ward/ICU", ["Ward", "ICU"])

        # Vitals
        temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, step=0.1)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, step=1)
        respiratory_rate = st.number_input("Respiratory Rate", min_value=5, max_value=60, step=1)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, step=1)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, step=1)
        spo2 = st.number_input("SpO₂ (%)", min_value=50, max_value=100, step=1)

        # Severity
        gcs = st.number_input("Glasgow Coma Scale (GCS)", min_value=3, max_value=15, step=1)
        oxygen_support = st.selectbox("Need Oxygen?", ["No", "Yes"])
        ventilation_needed = st.selectbox("Ventilation Needed?", ["No", "Yes"])
        bleeding = st.selectbox("Bleeding Present?", ["No", "Yes"])

        submitted = st.form_submit_button("🔮 Predict Outcome")

        if submitted:
            # Encode categorical values
            sex_val = 1 if sex == "M" else 0
            ward_val = 1 if ward_icu == "ICU" else 0
            oxygen_val = 1 if oxygen_support == "Yes" else 0
            ventilation_val = 1 if ventilation_needed == "Yes" else 0
            bleeding_val = 1 if bleeding == "Yes" else 0

            # Feature vector (must follow expected_features order)
            features = [[
                age, sex_val, ward_val,
                temperature, heart_rate, respiratory_rate,
                systolic_bp, diastolic_bp, spo2,
                gcs, oxygen_val, ventilation_val, bleeding_val
            ]]

            # Scale features
            features_scaled = scaler.transform(features)

            # Predict
            # Ensure prediction is a plain integer
prediction = model.predict(scaled_features)
prediction = int(prediction[0])   # take the first element and cast to int

# Decode label
outcome_label = label_encoder.inverse_transform([prediction])[0]


# ==============================
# CSV Upload
# ==============================
with st.expander("📂 CSV Upload"):
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)

        # Map categorical manually (must match training)
        df_uploaded["Sex"] = df_uploaded["Sex"].map({"M": 1, "F": 0})
        df_uploaded["WardType"] = df_uploaded["WardType"].map({"Ward": 0, "ICU": 1})
        df_uploaded["OxygenNeeded"] = df_uploaded["OxygenNeeded"].map({"No": 0, "Yes": 1})
        df_uploaded["VentilationNeeded"] = df_uploaded["VentilationNeeded"].map({"No": 0, "Yes": 1})
        df_uploaded["BleedingPresent"] = df_uploaded["BleedingPresent"].map({"No": 0, "Yes": 1})

        # Ensure column order
        df_uploaded = df_uploaded[expected_features]

        # Scale
        features_scaled = scaler.transform(df_uploaded)

        # Predict
        predictions = model.predict(features_scaled)
        df_uploaded["PredictedOutcome"] = label_encoder.inverse_transform(predictions)

        st.write("### 📊 Predictions")
        st.dataframe(df_uploaded)

        # Download predictions
        csv = df_uploaded.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
