import streamlit as st
import pandas as pd
import pickle
import time

# -------------------------------
# Load Pre-trained Models & Tools
# -------------------------------
with open("svmnew_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scallernew.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encodersnew.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# -------------------------------
# App Title
# -------------------------------
st.title("🩺 lassa Outcome Prediction App")
st.write("Upload patient data manually or via CSV to predict outcomes using the trained the model.")

# -------------------------------
# Input Columns
# -------------------------------
input_columns = [
    "Age", "Sex", "WardType", "Temperature_C", "HeartRate_bpm", 
    "RespRate_bpm", "SystolicBP_mmHg", "DiastolicBP_mmHg", "SpO2_%", 
    "GCS", "OxygenNeeded", "VentilationNeeded", "BleedingPresent"
]

categorical_columns = ["Sex", "WardType"]

# -------------------------------
# Manual Input Form
# -------------------------------
st.header("📝 Manual Patient Data Entry")

with st.form("manual_input_form"):
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    ward = st.selectbox("WardType", ["General", "ICU", "Emergency"])
    temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, step=0.1)
    hr = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, step=1)
    rr = st.number_input("Respiratory Rate (bpm)", min_value=5, max_value=60, step=1)
    sbp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, step=1)
    dbp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, step=1)
    spo2 = st.number_input("SpO₂ (%)", min_value=0, max_value=100, step=1)
    gcs = st.number_input("GCS", min_value=3, max_value=15, step=1)
    oxygen = st.selectbox("Oxygen Needed", [0, 1])
    ventilation = st.selectbox("Ventilation Needed", [0, 1])
    bleeding = st.selectbox("Bleeding Present", [0, 1])

    submit_manual = st.form_submit_button("🔮 Predict Outcome")

if submit_manual:
    # Prepare dataframe
    data = {
        "Age": [age], "Sex": [sex], "WardType": [ward], "Temperature_C": [temp],
        "HeartRate_bpm": [hr], "RespRate_bpm": [rr], "SystolicBP_mmHg": [sbp], 
        "DiastolicBP_mmHg": [dbp], "SpO2_%": [spo2], "GCS": [gcs],
        "OxygenNeeded": [oxygen], "VentilationNeeded": [ventilation], "BleedingPresent": [bleeding]
    }
    df = pd.DataFrame(data)

    # Encode categoricals
    for col in categorical_columns:
        df[col] = label_encoders[col].transform(df[col])

    # Scale and Predict with Spinner
    with st.spinner("⏳ Running prediction... Please wait."):
        time.sleep(2)  # simulate delay
        Xs = scaler.transform(df)
        pred = model.predict(Xs)[0]
        label = label_encoders["Outcome"].inverse_transform([pred])[0]

    st.success(f"✅ Predicted Outcome: **{label}**")

# -------------------------------
# CSV Upload
# -------------------------------
st.header("📂 Upload CSV for Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)

    # Check required columns
    missing_cols = [col for col in input_columns if col not in df_uploaded.columns]
    if missing_cols:
        st.error(f"❌ Missing columns in CSV: {missing_cols}")
    else:
        # Encode categoricals
        for col in categorical_columns:
            df_uploaded[col] = label_encoders[col].transform(df_uploaded[col])

        X = df_uploaded[input_columns]

        # Scale & predict with spinner
        with st.spinner("⏳ Processing CSV... Please wait."):
            time.sleep(2)  # simulate delay
            Xs = scaler.transform(X)
            preds = model.predict(Xs)
            df_uploaded["PredictedOutcome"] = label_encoders["Outcome"].inverse_transform(preds)

        st.success("✅ Predictions done. Preview:")
        st.dataframe(df_uploaded.head(10))

        # Option to download full predictions
        csv_out = df_uploaded.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv_out, "predicted_outcomes.csv", "text/csv")
