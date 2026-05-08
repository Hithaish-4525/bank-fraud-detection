import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="FraudShield AI", layout="wide")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'fraud_model.pkl')

st.title("🛡️ Bank Fraud Detection Dashboard")

# Load model
@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

st.sidebar.header("Transaction Input")
st.write("Adjust the values to see how the model reacts.")

# Let's create sliders for the most important features (V14, V17, V12 are usually top in Kaggle)
v14 = st.sidebar.slider("V14 (Risk Factor A)", -15.0, 5.0, 0.0)
v17 = st.sidebar.slider("V17 (Risk Factor B)", -15.0, 5.0, 0.0)
v12 = st.sidebar.slider("V12 (Risk Factor C)", -15.0, 5.0, 0.0)
amount = st.sidebar.number_input("Transaction Amount ($)", value=100.0)

if st.button("Check for Fraud"):
    # Create the full 30 features (mostly zeros for the demo)
    features = [0.0] * 30
    features[13] = v14 # V14
    features[16] = v17 # V17
    features[11] = v12 # V12
    features[29] = amount # Amount
    
    # Prepare for model
    df = pd.DataFrame([features], columns=model.feature_names_in_)
    
    # Predict
    prob = model.predict_proba(df)[0][1]
    
    # Results
    if prob > 0.5:
        st.error(f"🚨 HIGH RISK DETECTED! Probability: {prob:.2%}")
        st.warning("Recommended Action: Block Transaction & Alert Customer")
    else:
        st.success(f"✅ TRANSACTION SAFE. Probability: {prob:.2%}")
        st.info("Recommended Action: Proceed with Payment")

    # Feature Importance Visualization
    st.subheader("Why was this decision made?")
    st.write("In the Kaggle dataset, low negative values in V14 and V17 are strong indicators of fraud.")