import streamlit as st
import pandas as pd
import joblib

model = joblib.load("KNN_HeartDiseases.pkl")
scaler = joblib.load("Scalor.pkl")
expected_columns = joblib.load("InputCoulmn.pkl")


st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="centered"
)

st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #271814;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ff5718 !important;
        font-weight: 600 !important;
    }
    
    /* Text */
    .stMarkdown, label, p {
        color: #ffffff !important;
    }
    
    /* Input fields */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div > div > div {
        background-color: #3d2a24 !important;
        color: #ffffff !important;
        border: 1px solid #ff5718 !important;
        border-radius: 8px !important;
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background-color: #ff5718 !important;
    }
    
    /* Button */
    .stButton > button {
        background-color: #ff5718 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 48px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #cc4513 !important;
        transform: translateY(-2px) !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #1a3d2e !important;
        color: #4ade80 !important;
        border-radius: 8px !important;
        padding: 20px !important;
        border-left: 4px solid #22c55e !important;
    }
    
    .stError {
        background-color: #3d1a1a !important;
        color: #f87171 !important;
        border-radius: 8px !important;
        padding: 20px !important;
        border-left: 4px solid #ef4444 !important;
    }
    
    .stSuccess p, .stError p {
        color: inherit !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }
    
    /* Divider */
    hr {
        border-color: #ff5718 !important;
        opacity: 0.3 !important;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Heart Disease Prediction")
st.markdown("How to use this tool: enter the values exactly as shown on the medical report your doctor gave you (for example age, blood pressure, cholesterol), press Predict to see an estimated risk show the result to your doctor, this tool gives an estimate only and does not replace medical advice.")
st.markdown("Enter your health metrics below for risk assessment")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

with col2:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown("---")

if st.button("Predict Risk"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    st.markdown("<br>", unsafe_allow_html=True)
    if prediction == 1:
        st.error(
            "HIGH RISK: Consultation with a healthcare professional is recommended")
    else:
        st.success("LOW RISK: Continue maintaining a healthy lifestyle")
