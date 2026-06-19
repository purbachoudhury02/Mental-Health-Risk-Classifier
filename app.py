import streamlit as st
import pandas as pd
import pickle

# Load saved model and encoders
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

st.set_page_config(page_title="Mental Health Risk Classifier", layout="centered")

st.title("🧠 Mental Health Risk Classifier")
st.write("Enter your information below to get a risk assessment.")
st.write("---")

# Input fields organized into columns
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 28)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
    employment_status = st.selectbox("Employment Status", ["Student", "Employed", "Unemployed", "Self-Employed"])
    sleep_hours = st.slider("Sleep Hours per Night", 2.0, 10.0, 7.0)
    physical_activity = st.slider("Physical Activity (hours/week)", 0.0, 15.0, 3.0)
    screen_time = st.slider("Screen Time (hours/day)", 0.0, 16.0, 6.0)
    social_support = st.slider("Social Support Score (1-10)", 1, 10, 5)
    working_hours = st.slider("Working Hours per Week", 0, 80, 40)

with col2:
    work_stress = st.slider("Work Stress Level (1-10)", 1, 10, 5)
    academic_pressure = st.slider("Academic Pressure Level (1-10)", 1, 10, 5)
    job_satisfaction = st.slider("Job Satisfaction Score (1-10)", 1, 10, 5)
    financial_stress = st.slider("Financial Stress Level (1-10)", 1, 10, 5)
    anxiety_score = st.slider("Anxiety Score (1-10)", 1, 10, 5)
    depression_score = st.slider("Depression Score (1-10)", 1, 10, 5)
    stress_level = st.slider("General Stress Level (1-10)", 1, 10, 5)
    mood_swings = st.slider("Mood Swings Frequency (1-10)", 1, 10, 5)
    concentration_difficulty = st.slider("Concentration Difficulty (1-10)", 1, 10, 5)

st.write("---")
st.subheader("Mental Health History")

col3, col4 = st.columns(2)
with col3:
    panic_attack = st.selectbox("History of Panic Attacks", ["No", "Yes"])
    family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])
with col4:
    previous_diagnosis = st.selectbox("Previous Mental Health Diagnosis", ["No", "Yes"])
    therapy_history = st.selectbox("Therapy History", ["No", "Yes"])

substance_use = st.selectbox("Substance Use", ["No", "Yes"])

st.write("---")

if st.button("Predict Risk Level", type="primary"):

    # Build input dataframe matching training column order
    input_data = pd.DataFrame([{
        'age': age,
        'gender': encoders['gender'].transform([gender])[0],
        'marital_status': encoders['marital_status'].transform([marital_status])[0],
        'education_level': encoders['education_level'].transform([education_level])[0],
        'employment_status': encoders['employment_status'].transform([employment_status])[0],
        'sleep_hours': sleep_hours,
        'physical_activity_hours_per_week': physical_activity,
        'screen_time_hours_per_day': screen_time,
        'social_support_score': social_support,
        'work_stress_level': work_stress,
        'academic_pressure_level': academic_pressure,
        'job_satisfaction_score': job_satisfaction,
        'financial_stress_level': financial_stress,
        'working_hours_per_week': working_hours,
        'anxiety_score': anxiety_score,
        'depression_score': depression_score,
        'stress_level': stress_level,
        'mood_swings_frequency': mood_swings,
        'concentration_difficulty_level': concentration_difficulty,
        'panic_attack_history': 1 if panic_attack == "Yes" else 0,
        'family_history_mental_illness': 1 if family_history == "Yes" else 0,
        'previous_mental_health_diagnosis': 1 if previous_diagnosis == "Yes" else 0,
        'therapy_history': 1 if therapy_history == "Yes" else 0,
        'substance_use': 1 if substance_use == "Yes" else 0,
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    risk_labels = {0: "LOW RISK", 1: "MEDIUM RISK", 2: "HIGH RISK"}
    risk_colors = {0: "green", 1: "orange", 2: "red"}

    st.write("---")
    st.markdown(f"### Prediction: :{risk_colors[prediction]}[{risk_labels[prediction]}]")

    st.write("**Confidence breakdown:**")
    st.progress(float(probabilities[0]), text=f"Low Risk: {probabilities[0]:.1%}")
    st.progress(float(probabilities[1]), text=f"Medium Risk: {probabilities[1]:.1%}")
    st.progress(float(probabilities[2]), text=f"High Risk: {probabilities[2]:.1%}")

    if prediction == 2:
        st.warning("This is a model prediction, not a clinical diagnosis. If you're concerned about your mental health, please consider speaking with a mental health professional.")