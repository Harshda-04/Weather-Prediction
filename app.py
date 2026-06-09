import streamlit as st
import pickle
import numpy as np
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌦️",
    layout="centered"
)

st.title("🌦️ Weather Prediction AI App")
st.write("Predict weather based on trained ML model")

st.markdown("---")

# ---------------- LOAD MODEL ----------------
try:
    model = pickle.load(open("weather_model.pkl", "rb"))
    st.success("Model loaded successfully ✔")
except Exception as e:
    st.error("Model load failed ❌")
    st.stop()

# ---------------- INPUTS ----------------
st.subheader("📊 Enter Features")

humidity = st.number_input("💧 Humidity (%)", value=50.0)
wind_speed = st.number_input("🌬️ Wind Speed", value=10.0)
meanpressure = st.number_input("📊 Mean Pressure", value=1013.0)

today = datetime.now()

year = st.number_input("📅 Year", value=today.year)
month = st.number_input("📅 Month", value=today.month)
day = st.number_input("📅 Day", value=today.day)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict"):

    try:
        input_data = np.array([[
            humidity,
            wind_speed,
            meanpressure,
            year,
            month,
            day
        ]])

        prediction = model.predict(input_data)[0]

        st.subheader("📊 Result")

        st.success(f"🌡️ Predicted Value: {prediction:.2f}")

        # Interpretation
        if prediction >= 35:
            st.error("🔥 Very Hot Weather")
        elif prediction >= 25:
            st.success("🌤️ Normal Weather")
        elif prediction >= 15:
            st.info("🌥️ Cool Weather")
        else:
            st.warning("❄️ Cold Weather")

    except Exception as e:
        st.error("Prediction failed ❌ Check model/features order")
        st.write(e)