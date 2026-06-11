import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("vehicle_model.pkl")

# Title
st.title("Vehicle Failure Prediction")

# User Input
engine_temp = st.number_input("Engine Temperature")
oil_level = st.number_input("Oil Level")
battery_voltage = st.number_input("Battery Voltage")
mileage = st.number_input("Mileage")

# Prediction
if st.button("Predict"):

    data = np.array([
        [
            engine_temp,
            oil_level,
            battery_voltage,
            mileage
        ]
    ])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Vehicle Failure Predicted")
    else:
        st.success("Vehicle is Safe")
