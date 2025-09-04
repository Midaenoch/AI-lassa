# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
st.title("🏥 Lasssa Outcome Prediction App")

st.markdown(
    "This app predicts **Outcome (Positive/Negative)** using a trained model. "
    "Fill values manually or upload a CSV that matches the template (downloadable below)."
)

# Show model expectations (helpful)
try:
    st.info(f"Model expects {model.n_features_in_} input features. App will provide features in the order: {expected_features}")
except Exception:
    pass

# ----------------------------
# Manual entry: flexible inputs but normalized to training labels
# ----------------------------
with st.expander("✍️ Manual Entry"):
    with st.form("manual_form"):
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1)
        sex = st.selectbox("Sex", ["Male", "Female"])   # matches dataset labels
        ward_type = st.selectbox("Ward Type", ["Ward", "ICU"])
        temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, value=80, step=1)
        resp_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=5, max_value=60, value=18, step=1)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=120, step=1)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80, step=1)
        spo2 = st.number_input("SpO₂ (%)", min_value=50, max_value=100, value=97, step=1)
        gcs = st.number_input("Glasgow Coma Scale (GCS)", min_value=3, max_value=15, value=15, step=1)

        # For binary flags expose Yes/No (mapped to 1/0) — dataset stores 0/1
        oxygen_needed = st.radio("Oxygen Needed?", ("No", "Yes"), index=0)
        ventilation_needed = st.radio("Ventilation Needed?", ("No", "Yes"), index=0)
        bleeding_present = st.radio("Bleeding Present?", ("No", "Yes"), index=0)

        submit_manual = st.form_submit_button("🔮 Predict Outcome")

    if submit_manual:
        try:
            # Build DataFrame in expected order
            bin_map = {"Yes": 1, "No": 0}
            row = pd.DataFrame([[
                age,
                sex,
                ward_type,
                temperature,
                heart_rate,
                resp_rate,
                systolic_bp,
                diastolic_bp,
                spo2,
                gcs,
                bin_map[oxygen_needed],
                bin_map[ventilation_needed],
                bin_map[bleeding_present]
            ]], columns=expected_features)

            # Normalize strings just in case
            row["Sex"] = normalize_sex_series(row["Sex"])
            row["WardType"] = normalize_ward_series(row["WardType"])

            # Validate categories before transform
            bad = []
            # Sex
            allowed_sex = set(label_encoders["Sex"].classes_.tolist())
            if not set(row["Sex"].unique()).issubset(allowed_sex):
                bad.append(("Sex", set(row["Sex"].unique()) - allowed_sex))
            # WardType
            allowed_ward = set(label_encoders["WardType"].classes_.tolist())
            if not set(row["WardType"].unique()).issubset(allowed_ward):
                bad.append(("WardType", set(row["WardType"].unique()) - allowed_ward))
            if bad:
                st.error(f"Unexpected categories: {bad}. Update input to match training labels: {label_encoders['Sex'].classes_.tolist()}, {label_encoders['WardType'].classes_.tolist()}")
            else:
                # Encode
                row["Sex"] = label_encoders["Sex"].transform(row["Sex"])
                row["WardType"] = label_encoders["WardType"].transform(row["WardType"])

                # Scale and predict
                import time  # make sure this is at the top of your file

                with st.spinner("⏳ Running prediction... Please wait."):
                    time.sleep(2)  # simulate processing delay
                    Xs = scaler.transform(X)
                    pred = model.predict(Xs)[0]
                    label = label_encoders["Outcome"].inverse_transform([pred])[0]
                
                st.success(f"✅ Predicted Outcome: **{label}**")


# ----------------------------
# CSV upload: check + normalize + predict
# ----------------------------
with st.expander("📂 CSV Upload"):
    uploaded_file = st.file_uploader("Upload CSV (must contain exact headers)", type=["csv"])
    # Template
    template_df = pd.DataFrame(columns=expected_features)
    st.download_button("📥 Download CSV Template", template_df.to_csv(index=False).encode("utf-8"), "template.csv", "text/csv")

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            df_uploaded = None

        if df_uploaded is not None:
            # Normalize header whitespace
            df_uploaded.columns = [c.strip() for c in df_uploaded.columns]

            missing = set(expected_features) - set(df_uploaded.columns)
            if missing:
                st.error(f"Missing required columns: {missing}. Use the template provided.")
            else:
                # Work with a copy of only the expected features
                df_work = df_uploaded[expected_features].copy()

                # Normalize columns
                df_work["Sex"] = normalize_sex_series(df_work["Sex"])
                df_work["WardType"] = normalize_ward_series(df_work["WardType"])

                # Map binary columns robustly
                for col in ["OxygenNeeded", "VentilationNeeded", "BleedingPresent"]:
                    df_work[col] = safe_map_binary_series(df_work[col])

                # Check for NaNs produced by mapping
                nan_cols = {col: df_work[col].isna().sum() for col in ["OxygenNeeded", "VentilationNeeded", "BleedingPresent"]}
                if any(v > 0 for v in nan_cols.values()):
                    st.error(f"Found invalid binary values (could not map to 0/1): { {k:v for k,v in nan_cols.items() if v>0} } — fix CSV or use template.")
                else:
                    # Validate categorical values against encoder classes
                    bad_sex = set(df_work["Sex"].unique()) - set(label_encoders["Sex"].classes_.tolist())
                    bad_ward = set(df_work["WardType"].unique()) - set(label_encoders["WardType"].classes_.tolist())
                    if bad_sex or bad_ward:
                        st.error(f"Unexpected categorical values. Sex bad: {bad_sex}, WardType bad: {bad_ward}. Allowed Sex: {label_encoders['Sex'].classes_.tolist()}, WardType: {label_encoders['WardType'].classes_.tolist()}")
                    else:
                        # Encode
                        df_work["Sex"] = label_encoders["Sex"].transform(df_work["Sex"])
                        df_work["WardType"] = label_encoders["WardType"].transform(df_work["WardType"])

                        # Convert to float and check shape
                        X = df_work[expected_features].astype(float)
                        if X.shape[1] != model.n_features_in_:
                            st.error(f"Feature count mismatch: model expects {model.n_features_in_} features, CSV gives {X.shape[1]}.")
                        else:
                            # Scale & predict
                            Xs = scaler.transform(X)
                            preds = model.predict(Xs)
                            df_uploaded["PredictedOutcome"] = label_encoders["Outcome"].inverse_transform(preds)
                            st.success("✅ Predictions done. Preview:")
                            st.dataframe(df_uploaded.head(10))
                            st.download_button("📥 Download Predictions", df_uploaded.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
