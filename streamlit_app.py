import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Patient Data Collection", layout="wide")
st.title("🏥 Patient Data Collection App")

st.write("You can either **manually enter** patient records or **upload a CSV file**.")

# Initialize session state
if "records" not in st.session_state:
    st.session_state.records = []

# ==============================
# TAB 1: Manual Entry
# ==============================
tab1, tab2 = st.tabs(["✍️ Manual Entry", "📂 CSV Upload"])

with tab1:
    with st.form("patient_form"):
        st.subheader("👤 Demographics")
        age = st.number_input("Age (years)", min_value=0, max_value=120, step=1)
        sex = st.selectbox("Sex", ["M", "F"])

        st.subheader("🏥 Visit Context")
        admission_date = st.date_input("Admission Date", value=datetime.today())
        ward_icu = st.selectbox("Ward/ICU", ["Ward", "ICU"])
        outcome = st.selectbox("Outcome", ["Positive", "Negative"])

        st.subheader("🫁 Vital Signs")
        temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, step=0.1)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, step=1)
        respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=5, max_value=60, step=1)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, step=1)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, step=1)
        spo2 = st.number_input("SpO₂ (%)", min_value=50, max_value=100, step=1)

        st.subheader("⚠️ Clinical Severity")
        gcs = st.number_input("Glasgow Coma Scale (GCS)", min_value=3, max_value=15, step=1)
        oxygen_support = st.selectbox("Need Oxygen/Ventilation?", ["No", "Yes"])
        bleeding = st.selectbox("Presence of Bleeding?", ["No", "Yes"])

        submitted = st.form_submit_button("➕ Add Record")

        if submitted:
            record = {
                "Age": age,
                "Sex": sex,
                "AdmissionDate": admission_date,
                "WardICU": ward_icu,
                "Outcome": outcome,
                "Temperature": temperature,
                "HeartRate": heart_rate,
                "RespiratoryRate": respiratory_rate,
                "SystolicBP": systolic_bp,
                "DiastolicBP": diastolic_bp,
                "SpO2": spo2,
                "GCS": gcs,
                "OxygenSupport": oxygen_support,
                "Bleeding": bleeding,
            }
            st.session_state.records.append(record)
            st.success("✅ Record added successfully!")

# ==============================
# TAB 2: CSV Upload
# ==============================
with tab2:
    st.subheader("📂 Upload Patient Records CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            st.session_state.records.extend(df_uploaded.to_dict(orient="records"))
            st.success("✅ CSV data uploaded successfully!")
            st.write("Preview of uploaded data:")
            st.dataframe(df_uploaded.head())
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")

# ==============================
# Display Final Dataset
# ==============================
if st.session_state.records:
    st.write("### 📋 Collected Patient Records")
    df = pd.DataFrame(st.session_state.records)
    st.dataframe(df)

    # Download option
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download All Data as CSV", csv, "patient_records.csv", "text/csv")
