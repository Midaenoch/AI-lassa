import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ----------------------------
# Load ML components
# ----------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    model = joblib.load("svm_model.pkl")
    return scaler, label_encoder, model

scaler, label_encoder, model = load_artifacts()

st.set_page_config(page_title="Patient Outcome Predictor", layout="wide")
st.title("🏥 Patient Outcome Prediction App")

st.write("This app predicts **Outcome (Positive/Negative)** using a trained SVM model.")

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
        oxygen_support = st.selectbox("Need Oxygen/Ventilation?", ["No", "Yes"])
        bleeding = st.selectbox("Bleeding?", ["No", "Yes"])

        submitted = st.form_submit_button("🔮 Predict Outcome")

        if submitted:
            # Encode categorical values
            sex_val = 1 if sex == "M" else 0
            ward_val = 1 if ward_icu == "ICU" else 0
            oxygen_val = 1 if oxygen_support == "Yes" else 0
            bleeding_val = 1 if bleeding == "Yes" else 0

            # Feature vector
            features = [[
                age, sex_val, ward_val,
                temperature, heart_rate, respiratory_rate,
                systolic_bp, diastolic_bp, spo2,
                gcs, oxygen_val, bleeding_val
            ]]

            # Scale features
            features_scaled = scaler.transform(features)

            # Predict
            prediction = model.predict(features_scaled)[0]
            outcome_label = label_encoder.inverse_transform([prediction])[0]

            st.success(f"✅ Predicted Outcome: **{outcome_label}**")

# ==============================
# CSV Upload
# ==============================
with st.expander("📂 CSV Upload"):
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)

        # Map categorical manually (must match training)
        df_uploaded["Sex"] = df_uploaded["Sex"].map({"M": 1, "F": 0})
        df_uploaded["WardICU"] = df_uploaded["WardICU"].map({"Ward": 0, "ICU": 1})
        df_uploaded["OxygenSupport"] = df_uploaded["OxygenSupport"].map({"No": 0, "Yes": 1})
        df_uploaded["Bleeding"] = df_uploaded["Bleeding"].map({"No": 0, "Yes": 1})

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
