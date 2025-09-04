import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load ML components
# ----------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scallernew.pkl")
    label_encoders = joblib.load("label_encodersnew.pkl")  # dict: {"Sex": encoder, "WardType": encoder, "Outcome": encoder}
    model = joblib.load("svmnew_model.pkl")
    return scaler, label_encoders, model

scaler, label_encoders, model = load_artifacts()

# Define expected features
expected_features = [
    "Age", "Sex", "WardType",
    "Temperature_C", "HeartRate_bpm", "RespRate_bpm",
    "SystolicBP_mmHg", "DiastolicBP_mmHg", "SpO2_%",
    "GCS", "OxygenNeeded", "VentilationNeeded", "BleedingPresent"
]

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
        ward_type = st.selectbox("Ward Type", ["Ward", "ICU"])

        # Vitals
        temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, step=0.1)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, step=1)
        resp_rate = st.number_input("Respiratory Rate", min_value=5, max_value=60, step=1)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, step=1)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, step=1)
        spo2 = st.number_input("SpO₂ (%)", min_value=50, max_value=100, step=1)

        # Severity
        gcs = st.number_input("Glasgow Coma Scale (GCS)", min_value=3, max_value=15, step=1)
        oxygen_needed = st.selectbox("Oxygen Needed?", ["No", "Yes"])
        ventilation_needed = st.selectbox("Ventilation Needed?", ["No", "Yes"])
        bleeding_present = st.selectbox("Bleeding Present?", ["No", "Yes"])

        submitted = st.form_submit_button("🔮 Predict Outcome")

        if submitted:
            # Create dataframe
            input_data = pd.DataFrame([[
                age, sex, ward_type,
                temperature, heart_rate, resp_rate,
                systolic_bp, diastolic_bp, spo2,
                gcs, oxygen_needed, ventilation_needed, bleeding_present
            ]], columns=expected_features)

            # Encode categorical variables
            input_data["Sex"] = label_encoders["Sex"].transform(input_data["Sex"])
            input_data["WardType"] = label_encoders["WardType"].transform(input_data["WardType"])
            input_data["OxygenNeeded"] = input_data["OxygenNeeded"].map({"No": 0, "Yes": 1})
            input_data["VentilationNeeded"] = input_data["VentilationNeeded"].map({"No": 0, "Yes": 1})
            input_data["BleedingPresent"] = input_data["BleedingPresent"].map({"No": 0, "Yes": 1})

            # Scale
            scaled_features = scaler.transform(input_data)

            # Predict
            prediction = model.predict(scaled_features)[0]
            outcome_label = label_encoders["Outcome"].inverse_transform([prediction])[0]

            st.success(f"✅ Predicted Outcome: **{outcome_label}**")

# ==============================
# CSV Upload
# ==============================
with st.expander("📂 CSV Upload"):
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)

        # Ensure only expected features are used
        missing_cols = set(expected_features) - set(df_uploaded.columns)
        if missing_cols:
            st.error(f"Missing columns in CSV: {missing_cols}")
        else:
            # Encode categorical columns
            df_uploaded["Sex"] = label_encoders["Sex"].transform(df_uploaded["Sex"])
            df_uploaded["WardType"] = label_encoders["WardType"].transform(df_uploaded["WardType"])
            df_uploaded["OxygenNeeded"] = df_uploaded["OxygenNeeded"].map({"No": 0, "Yes": 1})
            df_uploaded["VentilationNeeded"] = df_uploaded["VentilationNeeded"].map({"No": 0, "Yes": 1})
            df_uploaded["BleedingPresent"] = df_uploaded["BleedingPresent"].map({"No": 0, "Yes": 1})

            # Scale
            scaled_features = scaler.transform(df_uploaded[expected_features])

            # Predict
            predictions = model.predict(scaled_features)
            df_uploaded["PredictedOutcome"] = label_encoders["Outcome"].inverse_transform(predictions)

            st.write("### 📊 Predictions")
            st.dataframe(df_uploaded)

            # Download predictions
            csv = df_uploaded.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
