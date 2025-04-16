import pickle
import streamlit as st
from  streamlit_option_menu import option_menu  
import pandas as pd
import numpy as np 
from PIL import Image

# Load model and scaler using pickle
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open('scaler.pkl', "rb") as f:
    scaler = pickle.load(f)

with open("heart_disease_model.pkl", "rb") as f:
    model1 = pickle.load(f)
with open("heart_scaler.pkl", "rb") as f:
    scaler1 = pickle.load(f)




with open("parkinsons_model.pkl", "rb") as f:
    model2 = pickle.load(f)

with open('parkinsons_scaler.pkl', "rb") as f:
    scaler2 = pickle.load(f)


with st.sidebar:
    selected = option_menu("Multiple Diseases Prediction System using ML",
                           ["Diabetes Disease Prediction",
                            "Heart Disease Prediction",
                            "Parkinsons Disease Prediction"],
                            icons=["activity","heart","person"],
                            default_index=0)



if selected == "Diabetes Disease Prediction":
    st.title("🩸 Diabetes Prediction using ML")
    st.write("Enter your health information below to predict your diabetes risk:")
    img = "photos/d3.jpg"
    st.image(img, caption="Diabetes disease prediction")

    # Input form
    col1,col2,col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)

    with col2:
        Glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)

    with col3:
        BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=140, value=70)

    with col1:
        SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    
    with col2:
        Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)

    with col3:
        BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)

    with col1:
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    
    with col2:
        Age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Predict
    if st.button("  🔍 Diabetic Test Predict"):
        try:
            input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1] * 100  # Convert to percentage

            st.markdown(f"### 🩸 Probability of Having Diabetes: **{prob:.2f}%**")
            if prob > 70:
                st.image("photos/high_risk.png", use_container_width=True)
                st.error("🔴 **High Risk!** 🚨 Please consult a doctor immediately.")
            elif prob > 40:
                st.image("photos/moderate_risk.png", use_container_width=True)
                st.warning("🟠 **Moderate Risk** ⚠️ Consider lifestyle changes and check-up.")
            
            else:
                st.image("photos/low_risk.png", use_container_width=True)
                st.success("🟢 **Low Risk** 🎉 Keep up your healthy habits!")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
        
#-----------------------------------------------------------------------------------------------------------------------
 
if(selected=="Heart Disease Prediction"):
    st.title(" ❤️ Heart Diseases Prediction")
    img = "photos/heart2.jpg"
    st.image(img,caption="Heart Prediction")

  
    st.write("Provide the following health parameters:")

# User input fields
    age = st.slider("Age", 18, 100, 50)
    sex_display = st.selectbox("Sex", ["Female", "Male"])
    sex = 0 if sex_display == "Female" else 1 # 0: Female, 1: Male

    cp_options = {
    "Typical angina (0)": 0,
    "Atypical angina (1)": 1,
    "Non-anginal pain (2)": 2,
    "Asymptomatic (3)": 3
      }
    cp_display = st.selectbox("Chest Pain Type", list(cp_options.keys()))
    cp = cp_options[cp_display]
    # cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
    thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3,4])
    thal = st.selectbox("Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)", [0, 1, 2,3])

# Prediction
    if st.button("🔍 Heart Test Predict"):
        try:
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal]])

            input_data_scaled = scaler1.transform(input_data)
            prediction1 = model1.predict(input_data_scaled)[0]
            prob1 = model1.predict_proba(input_data_scaled)[0][1] * 100
            st.markdown(f"### ❤️ Probability of Parkinson's Disease: **{prob1:.2f}%**")
            if prob1 > 70:
                st.image("photos/high_risk.png", use_container_width=True)
                st.error("🔴 **High Risk!** Consult a cardiologist immediately.")
            elif prob1 > 40:
                st.image("photos/moderate_risk.png", use_container_width=True)
                st.warning("🟠 **Moderate Risk** ⚠️ Consider lifestyle changes and check-up.")
            else:
                st.image("photos/low_risk.png", use_container_width=True)
                st.success("🟢 **Low Risk** Keep up the healthy lifestyle!")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
           


if(selected== "Parkinsons Disease Prediction"):
    
    st.title("🧠 Parkinson’s Disease Prediction Using ML")
    img = "photos/p1.jpg"
    st.image(img,caption="Parkinson's Prediction",width=700)

    feature_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
                 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
                 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']



# Create 3 columns
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

# Collect user input
    user_input = []

    for i, feature in enumerate(feature_names):
        with columns[i % 3]:
            val = st.text_input(feature)
            user_input.append(val)



#----------------------------------------------------------------------------------------------------------------------------------------
    
# Prediction
    if st.button(" 🔍 Parkinson's Test  Predict"):
        try:
            input_array = np.array(user_input).reshape(1, -1)
            input_scaled = scaler2.transform(input_array)
            prediction = model2.predict(input_scaled)
            prob2 =model2.predict_proba(input_scaled)[0][1] * 100
            st.markdown(f"### 🧠 Probability of Parkinson's Disease: **{prob2 :.2f}%**")
            if prob2 > 70:
                st.image("photos/high_risk.png", use_container_width=True)
                st.error("🔴 **High Risk!** Consult a neurologist immediately.")
                st.write("The input data suggests a high likelihood of Parkinson’s disease. Please consult a medical professional for further diagnosis.")
            elif prob2 > 40:
                st.image("photos/moderate_risk.png", use_container_width=True)
                st.warning("🟠 **Moderate Risk** ⚠️ Consider lifestyle changes and check-up.")
            else:
                st.image("photos/low_risk.png", use_container_width=True)
                st.success("✅ **Low Risk** Keep up the healthy lifestyle!")
                st.write("The input data does not indicate signs of Parkinson’s disease. Stay healthy!") 
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")


       

   
    
