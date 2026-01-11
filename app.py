import streamlit as st
import numpy as np
import joblib

# Load saved models
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca_model.pkl")
model = joblib.load("lr_model.pkl")

st.title("California Housing Price Predictor 🏡")
st.write("Enter the features to predict the median house value:")

# Collect all 8 feature inputs
MedInc = st.number_input("Median Income (MedInc)", min_value=0.0, value=3.0, step=0.1)
HouseAge = st.number_input("House Age", min_value=0.0, value=30.0, step=1.0)
AveRooms = st.number_input("Average Rooms per House", min_value=0.0, value=5.0, step=0.1)
AveBedrms = st.number_input("Average Bedrooms per House", min_value=0.0, value=1.0, step=0.1)
Population = st.number_input("Population", min_value=0.0, value=1000.0, step=1.0)
AveOccup = st.number_input("Average Occupancy", min_value=0.0, value=3.0, step=0.1)
Latitude = st.number_input("Latitude", min_value=0.0, value=34.0, step=0.01)
Longitude = st.number_input("Longitude", min_value=0.0, value=-118.0, step=0.01)

# Prepare input as array
input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

# Apply scaler and PCA
input_scaled = scaler.transform(input_data)
input_pca = pca.transform(input_scaled)

# Make prediction
prediction = model.predict(input_pca)

st.success(f"Predicted Median House Value: ${prediction[0]*100000:.2f}")
