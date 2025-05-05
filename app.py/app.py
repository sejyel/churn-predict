import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('best_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Customer Churn Prediction Dashboard")

st.write("Enter customer details to predict churn:")

# Example input fields (add more as per your model's features and encoding)
tenure = st.number_input("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 20000.0, 1000.0)

# Example for binary features (adjust as per your encoding)
gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

# One-hot encoded features (example)
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# Collect all features in the correct order as per your model
input_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'gender': 1 if gender == "Female" else 0,
    'Partner': 1 if partner == "Yes" else 0,
    'Dependents': 1 if dependents == "Yes" else 0,
    'PhoneService': 1 if phone_service == "Yes" else 0,
    'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
    # Add one-hot encoding for other features as needed
}

# For one-hot encoded features, you need to match the columns used in training
# For simplicity, you can use pd.get_dummies on a DataFrame with all possible categories

if st.button("Predict"):
    # Convert input_data to DataFrame
    df = pd.DataFrame([input_data])

    # If you have one-hot encoded features, make sure columns match training data
    # For demonstration, let's assume you already handled this

    # Scale numeric columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.transform(df[num_cols])

    # Predict
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    st.write(f"Churn Prediction: {'Yes' if pred==1 else 'No'} (Probability: {prob:.2%})")
