import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics

# Load trained model
with open("loan_approval_model.pkl", "rb") as f:
    model = pickle.load(f)

# Configure page
st.set_page_config(page_title="🏦 Bank Loan Approval Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: #2e86de;'>🏦 Bank Loan Approval Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Enter applicant details below to predict loan approval status</p>", unsafe_allow_html=True)
st.markdown("---")

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        deposit = st.selectbox("💰 Deposit", ["yes", "no"])
        poutcome = st.selectbox("📞 Previous Campaign Outcome", ["success", "failure", "other"])
        housing = st.selectbox("🏠 Has Housing Loan", ["yes", "no"])
        balance = st.number_input("💳 Account Balance", value=0)
        default = st.selectbox("❌ Has Credit Default", ["yes", "no"])

    with col2:
        education = st.selectbox("🎓 Education Level", ["primary", "secondary", "tertiary", "unknown"])
        loan = st.selectbox("📄 Has Personal Loan", ["yes", "no"])
        job = st.selectbox("👔 Job Type", [
            "unemployed", "student", "retired", "technician", "services",
            "self-employed", "management", "housemaid", "entrepreneur", "blue-collar"
        ])

    submitted = st.form_submit_button("🔍 Predict Loan Approval")

if submitted:
    # Encode inputs
    def binary_encode(val): return 1 if val == "yes" else 0

    deposit_encoded = binary_encode(deposit)
    housing_encoded = binary_encode(housing)
    default_encoded = binary_encode(default)
    loan_encoded = binary_encode(loan)

    poutcome_map = {"success": 1, "failure": -1, "other": 0}
    poutcome_encoded = poutcome_map[poutcome]

    education_map = {"primary": 0, "secondary": 1, "tertiary": 2, "unknown": 3}
    education_encoded = education_map[education]

    job_options = [
        "unemployed", "student", "retired", "technician", "services",
        "self-employed", "management", "housemaid", "entrepreneur", "blue-collar"
    ]
    job_onehot = [1 if job == j else 0 for j in job_options]

    # Combine into feature vector
    input_data = np.array([
        deposit_encoded, poutcome_encoded, housing_encoded, balance,
        default_encoded, education_encoded, *job_onehot, loan_encoded
    ]).reshape(1, -1)

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    confidence = probability[prediction] * 100

    st.markdown("---")
    if prediction == 1:
        st.success(f"✅ **Loan Approved!**\n\n**Confidence:** `{confidence:.2f}%`")
        st.balloons()
    else:
        st.error(f"❌ **Loan Not Approved**\n\n**Confidence:** `{confidence:.2f}%`")
        st.snow()

    # Summary table
    st.markdown("### 📋 Summary of Inputs")
    summary_data = {
        "Deposit": deposit,
        "Previous Outcome": poutcome,
        "Housing Loan": housing,
        "Balance": balance,
        "Credit Default": default,
        "Education": education,
        "Personal Loan": loan,
        "Job": job,
        "Prediction": "Approved" if prediction == 1 else "Not Approved",
        "Confidence (%)": f"{confidence:.2f}"
    }
    df_summary = pd.DataFrame(summary_data.items(), columns=["Feature", "Value"])
    st.table(df_summary)

    # Feature Importance
    st.markdown("### 📊 Feature Importance")
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        features = [
            "Deposit", "Poutcome", "Housing", "Balance", "Default", "Education",
            *[f"Job: {j}" for j in job_options], "Loan"
        ]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(features, importance)
        ax.set_xlabel("Importance Score")
        ax.set_title("Model Feature Importance")
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model.")

    # PDF Download
    st.markdown("### 📄 Download Prediction Report")

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))  # Safe for multilingual text
    styles = getSampleStyleSheet()
    elements = [Paragraph("🏦 Bank Loan Approval Prediction Report", styles['Title']), Spacer(1, 12)]

    for key, value in summary_data.items():
        elements.append(Paragraph(f"<b>{key}:</b> {value}", styles["Normal"]))
        elements.append(Spacer(1, 6))

    doc.build(elements)
    pdf = buffer.getvalue()

    st.download_button(
        label="📥 Download Report (PDF)",
        data=pdf,
        file_name="loan_prediction_report.pdf",
        mime="application/pdf"
    )
