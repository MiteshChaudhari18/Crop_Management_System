import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import os

# Load sample data
@st.cache_data
def load_data():
    # Ensures it works no matter where Streamlit is run from
    data_path = os.path.join(os.path.dirname(__file__), "crop_data.csv")
    data = pd.read_csv(data_path)
    return data

# Cache models
@st.cache_resource
def train_crop_model():
    model = RandomForestClassifier()
    # Dummy training data (you can update with real data later)
    X = [[22, 80, 0, 200], [24, 82, 1, 250]]
    y = ["Rice", "Wheat"]
    model.fit(X, y)
    return model

@st.cache_resource
def train_yield_model():
    model = LinearRegression()
    # Dummy training data (can be replaced with rainfall_yield.py logic)
    X = [[2015, 200], [2016, 220]]
    y = [2.5, 2.7]
    model.fit(X, y)
    return model

# Streamlit Page Config
st.set_page_config(page_title="Crop Management System", layout="wide")

# Page Title
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>Crop Management System</h1>
    <h4 style='text-align: center; color: gray;'>A smart assistant for better farming decisions</h4>
    <hr style="border:1px solid #bbb">
""", unsafe_allow_html=True)

# Load CSV dataset
data = load_data()

# Dataset Preview
with st.expander("📊 View Sample Dataset"):
    st.dataframe(data, use_container_width=True)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🌱 Crop Prediction",
    "🧪 Fertilizer Suggestion",
    "📈 Yield Estimator"
])

# ---------- Tab 1: Crop Prediction ----------
with tab1:
    st.markdown("### Enter Environmental Conditions")

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
            # Encode soil type
            soil_map = {"Sandy": 0, "Loamy": 1, "Clay": 2}
            soil_encoded = soil_map[soil_type]
            st.session_state["soil_type"] = soil_type

            # Predict crop
            model = train_crop_model()
            prediction = model.predict([[temp, humidity, soil_encoded, rainfall]])
            st.success(f"🌾 Recommended Crop: {prediction[0]}")

# ---------- Tab 2: Fertilizer Suggestion ----------
with tab2:
    st.markdown("### Fertilizer Suggestion Based on Soil Type")

    st.write("Based on selected soil type:")

    if st.button("Get Fertilizer Suggestion"):
        soil = st.session_state.get("soil_type", None)
        if soil == "Sandy":
            st.success("🧪 Recommended: Urea — Nitrogen-rich, ideal for sandy soil.")
        elif soil == "Loamy":
            st.success("🧪 Recommended: NPK — Balanced nutrients for loamy soil.")
        elif soil == "Clay":
            st.success("🧪 Recommended: Superphosphate — Great for clay soils.")
        else:
            st.warning("⚠️ Please use the Crop Prediction tab first to select a soil type.")

# ---------- Tab 3: Yield Estimator ----------
with tab3:
    st.markdown("### Predict Crop Yield")

    with st.form("yield_form"):
        col1, col2 = st.columns(2)
        with col1:
            rain_input = st.number_input("Rainfall (mm)", min_value=0.0)
        with col2:
            year_input = st.number_input("Year", min_value=2000, max_value=2100, step=1)

        yield_submit = st.form_submit_button("Estimate Yield")

        if yield_submit:
            model = train_yield_model()
            predicted_yield = model.predict([[year_input, rain_input]])
            st.success(f"🌱 Estimated Yield: {predicted_yield[0]:.2f} tons/hectare")

