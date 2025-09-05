# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time  # 👈 Added for delay

# ----------------------------
# Helpers
# ----------------------------
def safe_map_binary_series(s):
    """Map common binary representations to 0/1, else return NaN for invalids."""
    def map_val(v):
        if pd.isna(v):
            return np.nan
        v_str = str(v).strip().lower()
        if v_str in {"1", "yes", "y", "true", "t"}:
            return 1
        if v_str in {"0", "no", "n", "false", "f"}:
            return 0
        # numeric-like
        try:
            if float(v) == 1.0:
                return 1
            if float(v) == 0.0:
                return 0
        except Exception:
            pass
        return np.nan
    return s.map(map_val)

def normalize_sex_series(s):
    """Normalize sex to 'Male'/'Female' where possible."""
    def norm(v):
        if pd.isna(v):
            return v
        v_str = str(v).strip().lower()
        if v_str in {"m", "male"}:
            return "Male"
        if v_str in {"f", "female"}:
            return "Female"
        return str(v).strip()
    return s.map(norm)

def normalize_ward_series(s):
    """Normalize ward values to 'Ward'/'ICU'."""
    def norm(v):
        if pd.isna(v):
            return v
        v_str = str(v).strip().lower()
        if v_str in {"ward", "w"}:
            return "Ward"
        if "icu" in v_str:
            return "ICU"
        return str(v).strip()
    return s.map(norm)

# ----------------------------
# Load artifacts (safe path)
# ----------------------------
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    scaler_path = os.path.join(base, "scallernew.pkl")
    enc_path = os.path.join(base, "label_encodersnew.pkl")
    model_path = os.path.join(base, "svmnew_model.pkl")

    missing = []
    for p in (scaler_path, enc_path, model_path):
        if not os.path.exists(p):
            missing.append(os.path.basename(p))
    if missing:
        raise FileNotFoundError(f"Required files not found in app folder: {missing}")

    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(enc_path)  # expected dict: {"Sex":le, "WardType":le, "Outcome":le}
    model = joblib.load(model_path)
    return scaler, label_encoders, model

# Try loading artifacts; show friendly error if not present
try:
    scaler, label_encoders, model = load_artifacts()
except Exception as e:
    scaler = label_encoders = model = None
    st.error(f"Model files load error: {e}")
    st.stop()

# ----------------------------
# Expected features (exactly as in your CSV)
# ----------------------------
expected_features = [
    "Age", "Sex", "WardType",
    "Temperature_C", "HeartRate_bpm", "RespRate_bpm",
    "SystolicBP_mmHg", "DiastolicBP_mmHg", "SpO2_%",
    "GCS", "OxygenNeeded", "VentilationNeeded", "BleedingPresent"
]

st.set_page_config(page_title="Patient Outcome Predictor", layout="wide")
st.title("🏥 Lassa Outcome Prediction App")

st.markdown(
    "This app predicts **Outcome** using your trained model. "
    "Fill values manually or upload a CSV that matches the template (downloadable below)."
)

# ----------------------------
# Manual entry
# ----------------------------
with st.expander("✍️ Manual Entry"):
    with st.form("manual_form"):
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1)
        sex = st.selectbox("Sex", ["Male", "Female"])
        ward_type = st.selectbox("Ward Type", ["Ward", "ICU"])
        temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, value=80, step=1)
        resp_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=5, max_value=60, value=18, step=1)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=120, step=1)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80, step=1)
        spo2 = st.number_input("SpO₂ (%)", min_value=50, max_value=100, value=97, step=1)
        gcs = st.number_input("Glasgow Coma Scale (GCS)", min_value=3, max_value=15, value=15, step=1)

        oxygen_needed = st.radio("Oxygen Needed?", ("No", "Yes"), index=0)
        ventilation_needed = st.radio("Ventilation Needed?", ("No", "Yes"), index=0)
        bleeding_present = st.radio("Bleeding Present?", ("No", "Yes"), index=0)

        submit_manual = st.form_submit_button("🔮 Predict Outcome")

    if submit_manual:
        with st.spinner("⏳ Running prediction... Please wait."):
            time.sleep(3)  # 👈 simulate delay
            try:
                bin_map = {"Yes": 1, "No": 0}
                row = pd.DataFrame([[age, sex, ward_type, temperature, heart_rate,
                                     resp_rate, systolic_bp, diastolic_bp, spo2, gcs,
                                     bin_map[oxygen_needed], bin_map[ventilation_needed],
                                     bin_map[bleeding_present]]], columns=expected_features)

                row["Sex"] = normalize_sex_series(row["Sex"])
                row["WardType"] = normalize_ward_series(row["WardType"])

                row["Sex"] = label_encoders["Sex"].transform(row["Sex"])
                row["WardType"] = label_encoders["WardType"].transform(row["WardType"])

                Xs = scaler.transform(row.astype(float))
                pred = model.predict(Xs)[0]
                label = label_encoders["Outcome"].inverse_transform([pred])[0]
                st.success(f"✅ Predicted Outcome for lassa: **{label}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ----------------------------
# CSV upload
# ----------------------------
# CSV upload
# ----------------------------
with st.expander("📂 CSV Upload"):
    uploaded_file = st.file_uploader("Upload CSV (must contain exact headers)", type=["csv"])
    template_df = pd.DataFrame(columns=expected_features)
    st.download_button("📥 Download CSV Template", template_df.to_csv(index=False).encode("utf-8"),
                       "template.csv", "text/csv")

    if uploaded_file is not None:
        with st.spinner("⏳ Processing CSV... Please wait."):
            time.sleep(3)  # 👈 simulate delay
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                df_uploaded.columns = [c.strip() for c in df_uploaded.columns]

                df_uploaded["Sex"] = normalize_sex_series(df_uploaded["Sex"])
                df_uploaded["WardType"] = normalize_ward_series(df_uploaded["WardType"])
                for col in ["OxygenNeeded", "VentilationNeeded", "BleedingPresent"]:
                    df_uploaded[col] = safe_map_binary_series(df_uploaded[col])

                df_uploaded["Sex"] = label_encoders["Sex"].transform(df_uploaded["Sex"])
                df_uploaded["WardType"] = label_encoders["WardType"].transform(df_uploaded["WardType"])

                Xs = scaler.transform(df_uploaded[expected_features].astype(float))
                preds = model.predict(Xs)
                df_uploaded["PredictedOutcome"] = label_encoders["Outcome"].inverse_transform(preds)

                # ✅ Outbreak check
                pos_count = (df_uploaded["PredictedOutcome"] == "Positive").sum()
                neg_count = (df_uploaded["PredictedOutcome"] == "Negative").sum()

                st.success("✅ Predictions done. Preview:")
                st.dataframe(df_uploaded.head(10))
                st.download_button("📥 Download Predictions",
                                   df_uploaded.to_csv(index=False).encode("utf-8"),
                                   "predictions.csv", "text/csv")

                # ✅ Show outbreak status
                st.info(f"📊 Positive cases: {pos_count}, Negative cases: {neg_count}")
                if pos_count > neg_count:
                    st.error("🚨 Lassa Outbreak Declared! Positive cases exceed negative cases.")
                else:
                    st.success("✅ No outbreak detected. Negative cases are higher.")

            except Exception as e:
                st.error(f"CSV Prediction failed: {e}")

