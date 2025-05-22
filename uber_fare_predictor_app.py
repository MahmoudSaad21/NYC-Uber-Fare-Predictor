import streamlit as st
import pandas as pd
import folium
from datetime import datetime
from streamlit_folium import folium_static
from feature_transformer import transform_features
import joblib

# Load the trained model and encoder
@st.cache_resource
def load_model():
    model = joblib.load('uber_fare_predictor_catboost.pkl')
    encoder = joblib.load('label_encoder.pkl')
    return model, encoder

model, encoder = load_model()

# App title and description
st.title('🚕 Uber Fare Predictor')
st.markdown("""
Predict your Uber fare based on trip details. Enter your trip information below and click **Predict Fare**.
""")

# Input form
with st.form("fare_prediction_form"):
    st.header("Trip Details")

    col1, col2 = st.columns(2)

    with col1:
        pickup_lat = st.number_input("Pickup Latitude", value=40.7128, format="%.6f")
        pickup_lon = st.number_input("Pickup Longitude", value=-74.0060, format="%.6f")
        passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)

    with col2:
        dropoff_lat = st.number_input("Dropoff Latitude", value=40.7128, format="%.6f")
        dropoff_lon = st.number_input("Dropoff Longitude", value=-74.0060, format="%.6f")
        # Using date and time inputs separately
        pickup_date = st.date_input("Pickup Date", value=datetime.now())
        pickup_time = st.time_input("Pickup Time", value=datetime.now().time())

    submitted = st.form_submit_button("Predict Fare")

# When form is submitted
if submitted:
    # Combine date and time
    pickup_datetime = datetime.combine(pickup_date, pickup_time)

    # Create input dictionary
    input_data = {
        'pickup_datetime': pickup_datetime,
        'pickup_longitude': pickup_lon,
        'pickup_latitude': pickup_lat,
        'dropoff_longitude': dropoff_lon,
        'dropoff_latitude': dropoff_lat,
        'passenger_count': passenger_count
    }

    # Transform features
    features = transform_features(input_data)

    # Encode categorical features
    if 'dropoff_city' in features.columns:
        features['dropoff_city'] = encoder.transform(features['dropoff_city'])

    # Make prediction
    prediction = model.predict(features)[0]

    # Display results
    st.success(f"### Predicted Fare Amount: ${prediction:.2f}")

    # Show feature values
    with st.expander("Show feature details"):
        st.write("**Generated Features:**")
        st.dataframe(features)

        # Enhanced Map visualization with Folium
        st.write("**Trip Route:**")

        # Create a Folium map centered between the two points
        midpoint_lat = (pickup_lat + dropoff_lat) / 2
        midpoint_lon = (pickup_lon + dropoff_lon) / 2

        m = folium.Map(location=[midpoint_lat, midpoint_lon], zoom_start=12)

        # Add pickup marker (blue)
        folium.Marker(
            [pickup_lat, pickup_lon],
            popup="Pickup Location",
            icon=folium.Icon(color='blue', icon='arrow-up', prefix='fa')
        ).add_to(m)

        # Add dropoff marker (red)
        folium.Marker(
            [dropoff_lat, dropoff_lon],
            popup="Dropoff Location",
            icon=folium.Icon(color='red', icon='flag-checkered', prefix='fa')
        ).add_to(m)

        # Add a line between the points
        folium.PolyLine(
            locations=[[pickup_lat, pickup_lon], [dropoff_lat, dropoff_lon]],
            color='green',
            weight=3,
            opacity=0.7
        ).add_to(m)

        # Display the map in Streamlit
        folium_static(m, width=700, height=500)

# Add some information about the model
st.sidebar.header("About")
st.sidebar.info("""
This app predicts Uber fares using a CatBoost model trained on historical trip data.

**Key Features Used:**
- Pickup/Dropoff Locations
- Distance
- Time of Day
- Day of Week
- Location Characteristics

The model achieves an R² score of ~0.90 on test data.
""")
