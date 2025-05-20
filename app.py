import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and encoders
model = joblib.load("C:/Users/Haji Moinuddin/AppData/Local/Programs/Python/Python310/Scripts/xgb_accident_model.pkl")
label_encoders = joblib.load("C:/Users/Haji Moinuddin/AppData/Local/Programs/Python/Python310/Scripts/label_encoders.pkl")

# Helper: Handle unseen labels
def safe_label_transform(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        le.classes_ = np.append(le.classes_, value)
        return le.transform([value])[0]

# Function to preprocess and predict
def predict(input_data):
    input_df = pd.DataFrame([input_data])
    
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].apply(lambda x: safe_label_transform(le, x))
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df).max()
    return prediction, probability

# Streamlit App UI
st.title("🚧 Road Accident Risk Predictor")

st.markdown("Enter the road and weather conditions to predict accident risk level.")

# Input fields
city = st.selectbox("City", label_encoders['City'].classes_)
state = st.selectbox("State", label_encoders['State'].classes_)
weather = st.selectbox("Weather Condition", label_encoders['Weather_Condition'].classes_)
wind_dir = st.selectbox("Wind Direction", label_encoders['Wind_Direction'].classes_)
timezone = st.selectbox("Timezone", label_encoders['Timezone'].classes_)

start_lat = st.number_input("Start Latitude", value=33.0)
start_lng = st.number_input("Start Longitude", value=-84.0)
distance = st.number_input("Distance (mi)", value=0.5)
temp = st.number_input("Temperature (F)", value=70.0)
humidity = st.number_input("Humidity (%)", value=60.0)
pressure = st.number_input("Pressure (in)", value=30.0)
visibility = st.number_input("Visibility (mi)", value=10.0)
wind_speed = st.number_input("Wind Speed (mph)", value=5.0)
precip = st.number_input("Precipitation (in)", value=0.0)

# Boolean checkboxes
amenity = st.checkbox("Amenity")
bump = st.checkbox("Bump")
crossing = st.checkbox("Crossing")
junction = st.checkbox("Junction")
railway = st.checkbox("Railway")
traffic_signal = st.checkbox("Traffic Signal")

# Predict Button
if st.button("Predict"):
    sample = {
        'Start_Lat': start_lat,
        'Start_Lng': start_lng,
        'Distance(mi)': distance,
        'City': city,
        'County': 'Fulton',  # Replace or add as input if needed
        'State': state,
        'Timezone': timezone,
        'Temperature(F)': temp,
        'Wind_Chill(F)': temp - 5,
        'Humidity(%)': humidity,
        'Pressure(in)': pressure,
        'Visibility(mi)': visibility,
        'Wind_Direction': wind_dir,
        'Wind_Speed(mph)': wind_speed,
        'Precipitation(in)': precip,
        'Weather_Condition': weather,
        'Amenity': amenity,
        'Bump': bump,
        'Crossing': crossing,
        'Give_Way': False,
        'Junction': junction,
        'No_Exit': False,
        'Railway': railway,
        'Roundabout': False,
        'Station': False,
        'Stop': False,
        'Traffic_Calming': False,
        'Traffic_Signal': traffic_signal,
        'Sunrise_Sunset': 1,
        'Civil_Twilight': 1,
        'Nautical_Twilight': 1,
        'Astronomical_Twilight': 1
    }

    pred, prob = predict(sample)
    risk_map = {0: "Low", 1: "Moderate", 2: "High"}
    st.success(f"🚦 Predicted Risk Level: **{risk_map.get(pred, 'Unknown')}** with probability **{prob:.2f}**")
