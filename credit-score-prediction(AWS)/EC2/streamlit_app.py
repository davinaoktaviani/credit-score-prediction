import json
import os

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION        = os.environ.get("AWS_REGION", "us-east-1")


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(instance: dict) -> dict:
    runtime  = get_runtime_client()
    payload  = {"instances": [instance]}
    response = runtime.invoke_endpoint(
        EndpointName = ENDPOINT_NAME,
        ContentType  = "application/json",
        Accept       = "application/json",
        Body         = json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


st.set_page_config(page_title="Credit Score Predictor", layout="centered")
st.title("Credit Score Predictor")
st.markdown(
    "Prediksi **Credit Score** nasabah (Poor / Standard / Good) "
    "menggunakan ML yang di-deploy di **AWS SageMaker**."
)
st.caption(f"Endpoint: `{ENDPOINT_NAME}` | Region: `{REGION}`")
st.divider()

col1, col2 = st.columns(2)

with col1:
    month = st.selectbox("Month", [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ])
    age            = st.number_input("Age", min_value=18, max_value=100, value=30)
    occupation     = st.selectbox("Occupation", [
        "Architect","Developer","Doctor","Engineer","Entrepreneur",
        "Journalist","Lawyer","Manager","Mechanic","Media_Manager",
        "Musician","Scientist","Teacher","Writer","Accountant",
        "Nurse","Firefighter","Police_Officer","Other",
    ])
    annual_income  = st.number_input("Annual Income (USD)",         min_value=0.0, value=40000.0, step=500.0)
    monthly_salary = st.number_input("Monthly Inhand Salary (USD)", min_value=0.0, value=3000.0,  step=100.0)
    num_bank       = st.number_input("Num Bank Accounts",           min_value=0, max_value=20, value=3)
    num_cc         = st.number_input("Num Credit Cards",            min_value=0, max_value=20, value=3)
    interest_rate  = st.number_input("Interest Rate (%)",           min_value=0, max_value=100, value=15)
    num_loan       = st.number_input("Num of Loans",                min_value=0, max_value=20, value=3)
    delay_due      = st.number_input("Delay from Due Date (days)",  min_value=0, max_value=100, value=5)
    num_delayed    = st.number_input("Num of Delayed Payments",     min_value=0, max_value=50,  value=5)

with col2:
    changed_limit = st.number_input("Changed Credit Limit",          min_value=0.0, value=10.0, step=0.5)
    num_inquiries = st.number_input("Num Credit Inquiries",          min_value=0, max_value=20, value=3)
    credit_mix    = st.selectbox("Credit Mix",                       ["Bad", "Standard", "Good"])
    outstanding   = st.number_input("Outstanding Debt (USD)",        min_value=0.0, value=1000.0, step=50.0)
    utilization   = st.number_input("Credit Utilization Ratio (%)",  min_value=0.0, max_value=100.0, value=30.0)
    history_age   = st.number_input("Credit History Age (months)",   min_value=0, value=100)
    pay_min       = st.selectbox("Payment of Min Amount",            ["Yes", "No"])
    total_emi     = st.number_input("Total EMI per Month (USD)",     min_value=0.0, value=50.0,  step=5.0)
    amount_invest = st.number_input("Amount Invested Monthly (USD)", min_value=0.0, value=100.0, step=10.0)
    pay_behaviour = st.selectbox("Payment Behaviour", [
        "High_spent_Large_value_payments","High_spent_Medium_value_payments",
        "High_spent_Small_value_payments","Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments","Low_spent_Small_value_payments",
    ])
    monthly_balance = st.number_input("Monthly Balance (USD)", min_value=0.0, value=200.0, step=10.0)

st.divider()

if st.button("Predict Credit Score", use_container_width=True, type="primary"):
    instance = {
        "Month": month, "Age": age, "Occupation": occupation,
        "Annual_Income": annual_income, "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": num_bank, "Num_Credit_Card": num_cc,
        "Interest_Rate": interest_rate, "Num_of_Loan": num_loan,
        "Delay_from_due_date": delay_due, "Num_of_Delayed_Payment": num_delayed,
        "Changed_Credit_Limit": changed_limit, "Num_Credit_Inquiries": num_inquiries,
        "Credit_Mix": credit_mix, "Outstanding_Debt": outstanding,
        "Credit_Utilization_Ratio": utilization, "Credit_History_Age": history_age,
        "Payment_of_Min_Amount": pay_min, "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": amount_invest, "Payment_Behaviour": pay_behaviour,
        "Monthly_Balance": monthly_balance,
    }

    try:
        result = invoke_endpoint(instance)
    except NoCredentialsError:
        st.error(
            "AWS credentials tidak ditemukan. "
            "Pastikan EC2 punya IAM role dengan akses ke SageMaker."
        )
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        label  = result["labels"][0]
        probs  = result["probabilities"][0]
        classes = sorted(["Poor", "Standard", "Good"])

        if label == "Good":
            st.success(f"### Credit Score: **{label}**")
        elif label == "Standard":
            st.warning(f"### Credit Score: **{label}**")
        else:
            st.error(f"### Credit Score: **{label}**")
        
        import pandas as pd
        
        proba_df = pd.DataFrame({
            "Credit Score": classes,
            "Probability (%)": [round(p * 100, 2) for p in probs],
        
        }).sort_values("Probability (%)", ascending=False)
        
        st.markdown("#### Probability per Class")
        st.dataframe(proba_df, use_container_width=True, hide_index=True)
        st.bar_chart(proba_df.set_index("Credit Score"))
