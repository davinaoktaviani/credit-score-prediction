import os
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Credit Score Predictor",
    layout="centered",
)

@st.cache_resource
def load_model():
    model_path = os.environ.get(
        "MODEL_PATH",
        str(Path(__file__).resolve().parent / "artifacts" / "best_model.pkl"),
    )
    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan di: {model_path}")
        st.stop()
    return joblib.load(model_path)

model = load_model()

st.title("Credit Score Predictor")
st.markdown("Masukkan data nasabah untuk memprediksi **Credit Score** (Poor / Standard / Good).")
st.divider()

col1, col2 = st.columns(2)

with col1:
    month = st.selectbox(
        "Month",
        ["January","February","March","April","May","June",
         "July","August","September","October","November","December"],
    )
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    occupation = st.selectbox(
        "Occupation",
        ["Architect","Developer","Doctor","Engineer","Entrepreneur",
         "Journalist","Lawyer","Manager","Mechanic","Media_Manager",
         "Musician","Scientist","Teacher","Writer","Accountant","Nurse",
         "Firefighter","Police_Officer","Other"],
    )
    annual_income = st.number_input("Annual Income (USD)", min_value=0.0, value=40000.0, step=500.0)
    monthly_salary = st.number_input("Monthly Inhand Salary (USD)", min_value=0.0, value=3000.0, step=100.0)
    num_bank_accounts = st.number_input("Num Bank Accounts", min_value=0, max_value=20, value=3)
    num_credit_card = st.number_input("Num Credit Cards", min_value=0, max_value=20, value=3)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0, max_value=100, value=15)
    num_of_loan = st.number_input("Num of Loans", min_value=0, max_value=20, value=3)
    delay_from_due = st.number_input("Delay from Due Date (days)", min_value=0, max_value=100, value=5)
    num_delayed_payment = st.number_input("Num of Delayed Payments", min_value=0, max_value=50, value=5)

with col2:
    changed_credit_limit = st.number_input("Changed Credit Limit", min_value=0.0, value=10.0, step=0.5)
    num_credit_inquiries = st.number_input("Num Credit Inquiries", min_value=0, max_value=20, value=3)
    credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])
    outstanding_debt = st.number_input("Outstanding Debt (USD)", min_value=0.0, value=1000.0, step=50.0)
    credit_utilization = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0)
    credit_history_age = st.number_input("Credit History Age (months)", min_value=0, value=100)
    payment_of_min_amount = st.selectbox("Payment of Min Amount", ["Yes", "No"])
    total_emi = st.number_input("Total EMI per Month (USD)", min_value=0.0, value=50.0, step=5.0)
    amount_invested = st.number_input("Amount Invested Monthly (USD)", min_value=0.0, value=100.0, step=10.0)
    payment_behaviour = st.selectbox(
        "Payment Behaviour",
        ["High_spent_Large_value_payments","High_spent_Medium_value_payments",
         "High_spent_Small_value_payments","Low_spent_Large_value_payments",
         "Low_spent_Medium_value_payments","Low_spent_Small_value_payments"],
    )
    monthly_balance = st.number_input("Monthly Balance (USD)", min_value=0.0, value=200.0, step=10.0)

st.divider()

if st.button("Predict Credit Score", use_container_width=True, type="primary"):
    input_data = pd.DataFrame([{
        "Month": month,
        "Age": age,
        "Occupation": occupation,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_of_loan,
        "Delay_from_due_date": delay_from_due,
        "Num_of_Delayed_Payment": num_delayed_payment,
        "Changed_Credit_Limit": changed_credit_limit,
        "Num_Credit_Inquiries": num_credit_inquiries,
        "Credit_Mix": credit_mix,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization,
        "Credit_History_Age": credit_history_age,
        "Payment_of_Min_Amount": payment_of_min_amount,
        "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": amount_invested,
        "Payment_Behaviour": payment_behaviour,
        "Monthly_Balance": monthly_balance,
    }])

    try:
        prediction = model.predict(input_data)[0]
        proba      = model.predict_proba(input_data)[0]
        classes    = model.classes_
        
        if prediction == "Good":
            st.success(f"### Credit Score: **{prediction}**")
        elif prediction == "Standard":
            st.warning(f"### Credit Score: **{prediction}**")
        else:
            st.error(f"### Credit Score: **{prediction}**")
            
        st.markdown("#### Probability per Class")
        
        proba_df = pd.DataFrame({
            "Credit Score": classes,
            "Probability (%)": [round(p * 100, 2) for p in proba],
        }).sort_values("Probability (%)", ascending=False)
        
        st.dataframe(
            proba_df,
            use_container_width=True,
            hide_index=True
        )
    
    except Exception as e:
        st.error(f"Prediction error: {e}")