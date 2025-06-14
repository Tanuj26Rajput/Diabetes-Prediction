import pickle
import streamlit as st
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("imputer.pkl", "rb") as f:
    imputer = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Diabetes Prediction App")

feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
user_input = []

for feature in feature_names:
    value = st.number_input(f"Enter {feature}", min_value=0.0, format="%.2f")
    user_input.append(value)

if st.button("Predict"):
    input_data = np.array(user_input).reshape(1, -1)
    input_data = imputer.transform(input_data)
    input_data = scaler.transform(input_data)

    prediction = model.precit(input_data)[0][0]

    if prediction == 1:
        st.error("Prediction: Diabetic")
    else:
        st.error("Prediction: Not Diabetic")