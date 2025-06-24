import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Load sample data
@st.cache_data
def load_data():
    data = pd.read_csv("../data/crop_data.csv")
    return data

# Page config
st.set_page_config(page_title="Crop Management System", layout="wide")

# Centered title
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>Crop Management System</h1>
    <h4 style='text-align: center; color: gray;'>A smart assistant for better farming decisions</h4>
    <hr style="border:1px solid #bbb">
""", unsafe_allow_html=True)

# Show sample dataset
with st.expander("View Sample Dataset"):
    data = load_data()
    st.dataframe(data, use_container_width=True)

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "Crop Prediction",
    "Fertilizer Suggestion",
    "Yield Estimator"
])

# --------- Crop Prediction ---------
with tab1:
    st.markdown("### Enter Environmental Conditions")

    with st.container():
        with st.form("crop_form"):
            col1, col2 = st.columns(2)
            with col1:
                temp = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
                humidity = st.number_input("Humidity (%)", min_value=0.0, step=0.1)
            with col2:
                soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Clay"])
                rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)

            submit = st.form_submit_button("Predict Crop")

            if submit:
                soil_map = {"Sandy": 0, "Loamy": 1, "Clay": 2}
                soil_encoded = soil_map[soil_type]

                model = RandomForestClassifier()
                model.fit([[22, 80, 0, 200], [24, 82, 1, 250]], ["Rice", "Wheat"])
                prediction = model.predict([[temp, humidity, soil_encoded, rainfall]])

                st.success(f"Recommended Crop: {prediction[0]}")

# --------- Fertilizer Suggestion ---------
with tab2:
    st.markdown("### Fertilizer Suggestion Based on Soil Type")

    st.write("Based on selected soil type:")

    if st.button("Get Fertilizer Suggestion"):
        if soil_type == "Sandy":
            st.success("Recommended: Urea — Nitrogen-rich, ideal for sandy soil.")
        elif soil_type == "Loamy":
            st.success("Recommended: NPK — Balanced nutrients for loamy soil.")
        else:
            st.success("Recommended: Superphosphate — Great for clay soils.")

# --------- Yield Estimator ---------
with tab3:
    st.markdown("### Predict Crop Yield")

    with st.container():
        with st.form("yield_form"):
            col1, col2 = st.columns(2)
            with col1:
                rain_input = st.number_input("Rainfall (mm)", min_value=0.0)
            with col2:
                year_input = st.number_input("Year", min_value=2000, max_value=2100, step=1)

            yield_submit = st.form_submit_button("Estimate Yield")

            if yield_submit:
                model = LinearRegression()
                model.fit([[2015, 200], [2016, 220]], [2.5, 2.7])
                predicted_yield = model.predict([[year_input, rain_input]])
                st.success(f"Estimated Yield: {predicted_yield[0]:.2f} tons/hectare")
