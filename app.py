from src.pipeline.training import TrainPipeline
from src.utils.common import read_yaml_file
from src.pipeline.prediction import PredictionPipeline
from src.utils.exception_handler import MyException
import streamlit as st
import numpy as np
import requests

st.title("Vehicle Insurance Prediction")
st.write("Demo application for Vehicle Insurance Prediction.")
st.set_page_config(layout="wide")
API_URL = "http://localhost:8000/predict"

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age of vehicle owner", min_value=18, max_value=100, value=44)
        vintage = st.number_input("Vehicle Vintage (in years)", min_value=0, value=217)
        annual_premium = st.number_input("Annual Premium", min_value=0,value=40454)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        vehicle_age = st.selectbox("Vehicle Age", options=["< 1 Year", "1-2 Year", "> 2 Years"])
    with col2:
        vehicle_damage = st.selectbox("Has Vehicle Damage?", options=["Yes", "No"])
        driving_license = st.selectbox("Has Driving License?", options=["Yes", "No"])
        region_code = st.number_input("Region Code", min_value=0, max_value=1000, value=28)
        insured = st.selectbox("Is previously insured?", options=["Yes", "No"])
        policy_sales_channel = st.number_input("Policy Channel", min_value=1, value=26)
    submit_button = st.form_submit_button(label="Predict")
    if submit_button:
        payload = {
            "Age": age,
            "Vintage": vintage,
            "Annual_Premium": annual_premium,
            "Gender": gender,
            "Vehicle_Age": vehicle_age,
            "Vehicle_Damage": vehicle_damage,
            "Driving_License": 1 if driving_license == "Yes" else 0,
            "Region_Code": region_code,
            "Previously_Insured": 1 if insured == "Yes" else 0,
            "Policy_Sales_Channel": policy_sales_channel
        }

        with st.spinner("Calling prediction API..."):
            response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            if result["prediction"] == 1:
                st.success(f"Likely to buy insurance (Confidence: {result['probability']:.2f})")
            else:
                st.warning(f"Unlikely to buy insurance (Confidence: {1 - result['probability']:.2f})")
        else:
            st.error("Prediction API failed")
    else:
        st.write(":red[Please fill in the form and click Predict.]")

if st.button("Run training pipeline"):
    with st.spinner("Running training pipeline...", show_time=True):
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
    st.success("Training pipeline executed successfully.")