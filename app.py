"""
Streamlit app: Credit Risk Predictor

Loads the trained sklearn pipeline (preprocessing + model baked in) and lets
a user fill out a loan-application-style form. Predicts probability of
default and shows a color-coded verdict, plus a "Model performance" section
pulling from the metrics/plots saved by train.py.
"""

import json

import joblib
import pandas as pd
import streamlit as st

from mappings import (
    CHECKING_ACCOUNT_STATUS,
    CREDIT_HISTORY,
    EMPLOYMENT_SINCE,
    FOREIGN_WORKER,
    HAS_TELEPHONE,
    HOUSING,
    JOB,
    OTHER_DEBTORS,
    OTHER_INSTALLMENT_PLANS,
    PERSONAL_STATUS_SEX,
    PROPERTY,
    PURPOSE,
    SAVINGS_ACCOUNT,
    reverse_map,
)

st.set_page_config(
    page_title="Credit Risk Predictor", page_icon="💳", layout="centered"
)

# --------------------------------------------------------------------------
# Load model (cached across reruns)
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")


@st.cache_data
def load_metrics():
    with open("reports/metrics.json") as f:
        return json.load(f)


model = load_model()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("💳 Credit Risk Predictor")
st.markdown(
    """
Predicts whether a loan applicant is a **credit risk** (likely to default)
using a model trained on the classic **UCI Statlog German Credit** dataset.
Fill out the form below and click **Predict** to see the model's verdict.
"""
)

st.divider()

# --------------------------------------------------------------------------
# Form
# --------------------------------------------------------------------------
with st.form("credit_application_form"):

    st.subheader("Personal Info")
    col1, col2 = st.columns(2)
    with col1:
        personal_status_sex_label = st.selectbox(
            "Personal status / sex", list(reverse_map(PERSONAL_STATUS_SEX).keys())
        )
        age = st.slider("Age", min_value=18, max_value=80, value=35)
        num_dependents = st.selectbox("Number of dependents", [1, 2], index=0)
    with col2:
        foreign_worker_label = st.selectbox(
            "Foreign worker", list(reverse_map(FOREIGN_WORKER).keys())
        )
        has_telephone_label = st.selectbox(
            "Has telephone", list(reverse_map(HAS_TELEPHONE).keys())
        )
        job_label = st.selectbox("Job", list(reverse_map(JOB).keys()))

    st.subheader("Loan Details")
    col3, col4 = st.columns(2)
    with col3:
        purpose_label = st.selectbox("Purpose of loan", list(reverse_map(PURPOSE).keys()))
        credit_amount = st.number_input(
            "Credit amount (DM)", min_value=250, max_value=20000, value=2500, step=50
        )
        duration_months = st.slider(
            "Duration (months)", min_value=4, max_value=72, value=24
        )
    with col4:
        installment_rate_pct = st.selectbox(
            "Installment rate (% of disposable income)", [1, 2, 3, 4], index=2
        )
        other_installment_plans_label = st.selectbox(
            "Other installment plans", list(reverse_map(OTHER_INSTALLMENT_PLANS).keys())
        )
        existing_credits_count = st.selectbox(
            "Existing credits at this bank", [1, 2, 3, 4], index=0
        )

    st.subheader("Financial History")
    col5, col6 = st.columns(2)
    with col5:
        checking_account_status_label = st.selectbox(
            "Checking account status", list(reverse_map(CHECKING_ACCOUNT_STATUS).keys())
        )
        savings_account_label = st.selectbox(
            "Savings account", list(reverse_map(SAVINGS_ACCOUNT).keys())
        )
        credit_history_label = st.selectbox(
            "Credit history", list(reverse_map(CREDIT_HISTORY).keys())
        )
    with col6:
        employment_since_label = st.selectbox(
            "Employment since", list(reverse_map(EMPLOYMENT_SINCE).keys())
        )
        other_debtors_label = st.selectbox(
            "Other debtors / guarantors", list(reverse_map(OTHER_DEBTORS).keys())
        )
        property_label = st.selectbox("Property", list(reverse_map(PROPERTY).keys()))

    with st.expander("Housing & residence"):
        housing_label = st.selectbox("Housing", list(reverse_map(HOUSING).keys()))
        residence_since = st.selectbox(
            "Years at current residence", [1, 2, 3, 4], index=1
        )

    submitted = st.form_submit_button("Predict", use_container_width=True)

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
if submitted:
    row = {
        "checking_account_status": reverse_map(CHECKING_ACCOUNT_STATUS)[
            checking_account_status_label
        ],
        "duration_months": duration_months,
        "credit_history": reverse_map(CREDIT_HISTORY)[credit_history_label],
        "purpose": reverse_map(PURPOSE)[purpose_label],
        "credit_amount": credit_amount,
        "savings_account": reverse_map(SAVINGS_ACCOUNT)[savings_account_label],
        "employment_since": reverse_map(EMPLOYMENT_SINCE)[employment_since_label],
        "installment_rate_pct": installment_rate_pct,
        "personal_status_sex": reverse_map(PERSONAL_STATUS_SEX)[
            personal_status_sex_label
        ],
        "other_debtors": reverse_map(OTHER_DEBTORS)[other_debtors_label],
        "residence_since": residence_since,
        "property": reverse_map(PROPERTY)[property_label],
        "age": age,
        "other_installment_plans": reverse_map(OTHER_INSTALLMENT_PLANS)[
            other_installment_plans_label
        ],
        "housing": reverse_map(HOUSING)[housing_label],
        "existing_credits_count": existing_credits_count,
        "job": reverse_map(JOB)[job_label],
        "num_dependents": num_dependents,
        "has_telephone": reverse_map(HAS_TELEPHONE)[has_telephone_label],
        "foreign_worker": reverse_map(FOREIGN_WORKER)[foreign_worker_label],
    }

    X = pd.DataFrame([row])
    proba_default = model.predict_proba(X)[0, 1]
    pred = model.predict(X)[0]

    st.divider()
    st.subheader("Result")

    pct = proba_default * 100
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Predicted default probability", f"{pct:.1f}%")
    with c2:
        if pred == 1:
            st.error("🔴 High Risk — Likely Declined")
        else:
            st.success("🟢 Low Risk — Likely Approved")

    st.progress(min(max(proba_default, 0.0), 1.0))

st.divider()

# --------------------------------------------------------------------------
# Model performance section
# --------------------------------------------------------------------------
with st.expander("📊 Model performance"):
    st.markdown(
        "Comparison of all trained variants on the held-out test set "
        "(80/20 stratified split). The winning variant (highest ROC-AUC) "
        "was used to power predictions above."
    )
    metrics = load_metrics()
    metrics_df = pd.DataFrame(metrics)
    display_df = metrics_df[
        ["variant", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ].rename(
        columns={
            "variant": "Variant",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "roc_auc": "ROC-AUC",
        }
    )
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image("reports/confusion_matrix.png", caption="Confusion Matrix")
    with img_col2:
        st.image("reports/roc_curve.png", caption="ROC Curve")

st.caption(
    "Dataset: UCI Statlog (German Credit Data). Built with scikit-learn, "
    "imbalanced-learn, and Streamlit."
)
