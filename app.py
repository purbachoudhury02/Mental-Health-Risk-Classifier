import pandas as pd
import pickle
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mental Health Risk Classifier",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND ENCODERS
# ============================================================

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


# ============================================================
# TITLE / INTRODUCTION
# ============================================================

st.title("🧠 Mental Health Risk Classifier")

st.markdown(
    """
    ### An ML-based exploration of mental health risk patterns

    Mental health can be influenced by many interconnected factors,
    including lifestyle, academic and workplace pressure, social
    support, and personal history.

    This project uses a **Random Forest classification model** to
    analyze these factors and estimate whether a given combination
    of characteristics falls into a **Low, Medium, or High Risk**
    category based on patterns learned from the training dataset.

    > **Important:** This application is an educational machine-learning
    > project. Its output is a model prediction and **not a medical
    > diagnosis or clinical assessment**.
    """
)

st.divider()


# ============================================================
# WHY THIS PROJECT?
# ============================================================

st.header("💡 About the Model")

st.markdown(
    """
    A simple prediction such as **"High Risk"** does not tell us much
    unless we understand what the model is actually looking at.

    The purpose of this project is therefore not only to generate a
    prediction, but also to demonstrate how **multiple features can
    be combined and processed by a machine-learning classifier**.

    Instead of looking at one factor in isolation, the model receives
    information across several dimensions:

    - 🛌 **Lifestyle:** sleep, physical activity and screen time
    - 🎓 **Academic / Work:** academic pressure, work stress,
      working hours and job satisfaction
    - 💭 **Psychological indicators:** anxiety, depression, stress,
      mood swings and concentration difficulty
    - 👥 **Social factors:** social support
    - 📋 **Personal history:** previous diagnosis, therapy history,
      family history and panic attacks
    - 💰 **Financial factors:** financial stress

    These features are considered together by the trained model when
    producing its classification.
    """
)

st.divider()


# ============================================================
# HOW THE MODEL WORKS
# ============================================================

st.header("⚙️ How does the prediction work?")

st.markdown(
    """
    The prediction follows a simple machine-learning pipeline:
    """
)

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.subheader("1️⃣ Input")
    st.write(
        "The user provides information across 24 features "
        "representing demographic, lifestyle, academic, work, "
        "psychological and historical factors."
    )

with step2:
    st.subheader("2️⃣ Encoding")
    st.write(
        "Categorical information such as gender, education and "
        "employment status is converted into numerical values "
        "using the encoders saved during model training."
    )

with step3:
    st.subheader("3️⃣ Random Forest")
    st.write(
        "The resulting feature set is passed to the trained "
        "Random Forest classifier, which uses patterns learned "
        "from the training data."
    )

with step4:
    st.subheader("4️⃣ Prediction")
    st.write(
        "The classifier assigns the input to one of three "
        "learned categories: Low Risk, Medium Risk or High Risk."
    )

st.info(
    "The model does not apply a manually written rule such as "
    "'high anxiety = high risk'. Instead, the Random Forest "
    "learns patterns and relationships between features from "
    "the training data."
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.header("📝 Enter information for the model")

st.markdown(
    """
    The following information is used to create the feature vector
    that is passed to the trained classifier.

    **Tip:** There is no single input that automatically determines
    the final result. The model considers the combination of the
    supplied features.
    """
)


# ============================================================
# DEMOGRAPHIC + LIFESTYLE INFORMATION
# ============================================================

st.subheader("👤 Demographic & Lifestyle Factors")

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        18,
        70,
        28
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    education_level = st.selectbox(
        "Education Level",
        ["High School", "Bachelor", "Master", "PhD"]
    )

    employment_status = st.selectbox(
        "Employment Status",
        ["Student", "Employed", "Unemployed", "Self-Employed"]
    )

with col2:

    sleep_hours = st.slider(
        "Sleep Hours per Night",
        2.0,
        10.0,
        7.0
    )

    physical_activity = st.slider(
        "Physical Activity (hours/week)",
        0.0,
        15.0,
        3.0
    )

    screen_time = st.slider(
        "Screen Time (hours/day)",
        0.0,
        16.0,
        6.0
    )

    social_support = st.slider(
        "Social Support Score (1-10)",
        1,
        10,
        5
    )

    working_hours = st.slider(
        "Working Hours per Week",
        0,
        80,
        40
    )


with st.expander("ℹ️ Why are these factors included?"):

    st.write(
        """
        These variables provide context about the individual's
        demographic and everyday lifestyle patterns.

        For example, sleep, physical activity, screen time and
        social support can provide additional information about
        the overall pattern represented by an input.

        The model does not interpret these values independently.
        They are supplied together with the other features.
        """
    )


# ============================================================
# WORK / ACADEMIC / FINANCIAL FACTORS
# ============================================================

st.subheader("🎓 Work, Academic & Financial Factors")

col3, col4 = st.columns(2)

with col3:

    work_stress = st.slider(
        "Work Stress Level (1-10)",
        1,
        10,
        5
    )

    academic_pressure = st.slider(
        "Academic Pressure Level (1-10)",
        1,
        10,
        5
    )

    job_satisfaction = st.slider(
        "Job Satisfaction Score (1-10)",
        1,
        10,
        5
    )

with col4:

    financial_stress = st.slider(
        "Financial Stress Level (1-10)",
        1,
        10,
        5
    )

    st.markdown(
        """
        **Why are these included?**

        Academic pressure, workplace stress, financial stress,
        working hours and job satisfaction represent different
        forms of environmental pressure that may appear in the
        dataset's learned patterns.
        """
    )


# ============================================================
# PSYCHOLOGICAL INDICATORS
# ============================================================

st.subheader("🧠 Psychological Indicators")

st.markdown(
    """
    These variables represent self-reported or dataset-provided
    measures related to psychological experiences.
    """
)

col5, col6 = st.columns(2)

with col5:

    anxiety_score = st.slider(
        "Anxiety Score (1-10)",
        1,
        10,
        5
    )

    depression_score = st.slider(
        "Depression Score (1-10)",
        1,
        10,
        5
    )

    stress_level = st.slider(
        "General Stress Level (1-10)",
        1,
        10,
        5
    )

with col6:

    mood_swings = st.slider(
        "Mood Swings Frequency (1-10)",
        1,
        10,
        5
    )

    concentration_difficulty = st.slider(
        "Concentration Difficulty (1-10)",
        1,
        10,
        5
    )


with st.expander("ℹ️ Why are these variables important to the model?"):

    st.write(
        """
        Anxiety, depression, stress, mood swings and concentration
        difficulty provide information about psychological patterns
        represented in the dataset.

        These features are especially important to understand as a
        group: the model receives all of them simultaneously rather
        than using one score as a direct decision rule.
        """
    )


# ============================================================
# MENTAL HEALTH HISTORY
# ============================================================

st.subheader("📋 Mental Health History")

col7, col8 = st.columns(2)

with col7:

    panic_attack = st.selectbox(
        "History of Panic Attacks",
        ["No", "Yes"]
    )

    family_history = st.selectbox(
        "Family History of Mental Illness",
        ["No", "Yes"]
    )

with col8:

    previous_diagnosis = st.selectbox(
        "Previous Mental Health Diagnosis",
        ["No", "Yes"]
    )

    therapy_history = st.selectbox(
        "Therapy History",
        ["No", "Yes"]
    )

substance_use = st.selectbox(
    "Substance Use",
    ["No", "Yes"]
)


with st.expander("ℹ️ Why does personal history matter?"):

    st.write(
        """
        Personal and family history provide additional context that
        can be present in the patterns learned by the model.

        These variables are not interpreted as medical conclusions.
        They are simply features that were part of the dataset used
        to train the classifier.
        """
    )


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Risk Level",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # BUILD INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        "age": age,

        "gender":
            encoders["gender"].transform([gender])[0],

        "marital_status":
            encoders["marital_status"].transform(
                [marital_status]
            )[0],

        "education_level":
            encoders["education_level"].transform(
                [education_level]
            )[0],

        "employment_status":
            encoders["employment_status"].transform(
                [employment_status]
            )[0],

        "sleep_hours":
            sleep_hours,

        "physical_activity_hours_per_week":
            physical_activity,

        "screen_time_hours_per_day":
            screen_time,

        "social_support_score":
            social_support,

        "work_stress_level":
            work_stress,

        "academic_pressure_level":
            academic_pressure,

        "job_satisfaction_score":
            job_satisfaction,

        "financial_stress_level":
            financial_stress,

        "working_hours_per_week":
            working_hours,

        "anxiety_score":
            anxiety_score,

        "depression_score":
            depression_score,

        "stress_level":
            stress_level,

        "mood_swings_frequency":
            mood_swings,

        "concentration_difficulty_level":
            concentration_difficulty,

        "panic_attack_history":
            1 if panic_attack == "Yes" else 0,

        "family_history_mental_illness":
            1 if family_history == "Yes" else 0,

        "previous_mental_health_diagnosis":
            1 if previous_diagnosis == "Yes" else 0,

        "therapy_history":
            1 if therapy_history == "Yes" else 0,

        "substance_use":
            1 if substance_use == "Yes" else 0,

    }])


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]


    # --------------------------------------------------------
    # RISK LABELS
    # --------------------------------------------------------

    risk_labels = {
        0: "LOW RISK",
        1: "MEDIUM RISK",
        2: "HIGH RISK"
    }


    risk_descriptions = {

        0: """
        The model classified this particular combination of
        features as **Low Risk** according to the patterns it
        learned from the training data.

        This does not mean that the individual has no mental
        health concerns. It simply means that the supplied
        feature combination was classified into the Low Risk
        category by this model.
        """,

        1: """
        The model classified this particular combination of
        features as **Medium Risk** according to the patterns
        it learned from the training data.

        This indicates that the feature combination falls into
        the model's intermediate classification category.
        It should not be interpreted as a clinical assessment.
        """,

        2: """
        The model classified this particular combination of
        features as **High Risk** according to the patterns
        it learned from the training data.

        This does not mean that the model has diagnosed a
        mental health condition. It means that the supplied
        feature combination resembles patterns associated with
        the High Risk class in the training data.
        """
    }


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header("📊 Model Result")

    if prediction == 0:

        st.success(
            f"### {risk_labels[prediction]}"
        )

    elif prediction == 1:

        st.warning(
            f"### {risk_labels[prediction]}"
        )

    else:

        st.error(
            f"### {risk_labels[prediction]}"
        )


    st.markdown(
        risk_descriptions[prediction]
    )


    # ========================================================
    # CONFIDENCE BREAKDOWN
    # ========================================================

    st.subheader("📈 Prediction Probability Breakdown")

    st.markdown(
        """
        The values below show the probability estimates produced
        by the Random Forest model for each class.

        These numbers represent the model's confidence distribution,
        **not the probability that a person actually has a mental
        health condition**.
        """
    )


    prob_col1, prob_col2, prob_col3 = st.columns(3)


    with prob_col1:

        st.metric(
            "Low Risk",
            f"{probabilities[0]:.1%}"
        )

        st.progress(
            float(probabilities[0])
        )


    with prob_col2:

        st.metric(
            "Medium Risk",
            f"{probabilities[1]:.1%}"
        )

        st.progress(
            float(probabilities[1])
        )


    with prob_col3:

        st.metric(
            "High Risk",
            f"{probabilities[2]:.1%}"
        )

        st.progress(
            float(probabilities[2])
        )


    # ========================================================
    # WHY DID THE MODEL GIVE THIS RESULT?
    # ========================================================

    st.subheader("🔍 Why did the model give this result?")

    st.markdown(
        """
        A common misunderstanding with machine-learning models is
        that the model follows a simple rule such as:

        > "If anxiety is high → High Risk."

        **That is not how this Random Forest classifier works.**

        Instead, the model evaluates the **combination of all the
        supplied features** and compares that combination against
        patterns learned from the training data.

        Therefore, changing one value may or may not change the
        prediction because the other features are still contributing
        to the overall classification.
        """
    )


    # ========================================================
    # CURRENT INPUT SUMMARY
    # ========================================================

    with st.expander("🔎 View the values given to the model"):

        display_data = pd.DataFrame({

            "Feature": [
                "Age",
                "Sleep Hours",
                "Physical Activity",
                "Screen Time",
                "Social Support",
                "Work Stress",
                "Academic Pressure",
                "Job Satisfaction",
                "Financial Stress",
                "Working Hours",
                "Anxiety",
                "Depression",
                "Stress",
                "Mood Swings",
                "Concentration Difficulty",
                "Panic Attack History",
                "Family History",
                "Previous Diagnosis",
                "Therapy History",
                "Substance Use"
            ],

            "Value": [
                age,
                sleep_hours,
                physical_activity,
                screen_time,
                social_support,
                work_stress,
                academic_pressure,
                job_satisfaction,
                financial_stress,
                working_hours,
                anxiety_score,
                depression_score,
                stress_level,
                mood_swings,
                concentration_difficulty,
                panic_attack,
                family_history,
                previous_diagnosis,
                therapy_history,
                substance_use
            ]

        })

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # LIMITATIONS
    # ========================================================

    st.subheader("⚠️ Important limitations")

    st.markdown(
        """
        **This project is an educational demonstration of machine
        learning, not a medical tool.**

        Some important limitations are:

        - The model can only learn patterns present in its training data.
        - A prediction can be incorrect.
        - Model probability is not the same thing as clinical probability.
        - Correlation in the dataset does not necessarily mean causation.
        - The model does not understand an individual's personal
          circumstances or context.
        - The prediction should not be used to make medical decisions.

        If someone is experiencing significant mental-health concerns,
        a qualified mental-health professional should be consulted
        instead of relying on this model.
        """
    )