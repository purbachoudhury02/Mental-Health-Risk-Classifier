# MENTAL HEALTH RISK CLASSIFIER:

An end-to-end machine learning pipeline that predicts mental health risk level 
(Low / Medium / High) based on lifestyle, workplace, and psychological factors.

## Project Overview:

Mental health risk is predicted using 24 features including sleep hours, 
work stress, screen time, social support, and psychological indicators.
This is a 3-class classification problem.

## Dataset:

- 25,000 records, 25 columns
- Source: Kaggle
- Target variable: mental_health_risk (0 = Low, 1 = Medium, 2 = High)
- No missing values

##  Project structure

| File | Description |
|------|-------------|
| `MHRCdataexploration.ipynb` | Dataset overview and first look |
| `MHRCvisualisation.ipynb` | Exploratory Data Analysis (EDA) with charts and key findings |
| `MHRCpreprocessing.ipynb` | Categorical encoding, feature scaling, and custom feature engineering |
| `MHRCmodeling.ipynb` | Model training, evaluation, and feature importance analysis |
| `MHRCsqlAnalysis.ipynb` | SQL queries and analysis on the dataset |
| `MHRCdemo.ipynb` | End-to-end demo with serialization pipeline and prediction on new input |
| `app.py` | Interactive Streamlit web application |

## Results

| Model | Accuracy | High-risk Recall |
|--------|----------|------------------|
| Logistic Regression | 74.84% | 63.00% |
| Random Forest | 97.38% | 88.00% |

## Top Predictive Features:

1.stress_level / composite_stress_score: Acts as the primary environmental trigger; compounding workplace, academic, and financial pressures forms a highly accurate baseline representation of cumulative psychological load.

2.sleep_hours: The foundational physiological metric for cognitive resilience. The pipeline verified a severe gradient change, proving that high-risk individuals average nearly 2 hours less sleep per night than low-risk profiles.

3.anxiety_score / depression_score: High-weight clinical baseline metrics that scale aggressively as individuals cross dangerous psychological thresholds.

 ## Demo App:
 Run locally with-
 pip install streamlit
 streamlit run app.py
 ## Tools & Tech Stack
Language & Core: Python, Jupyter Notebook

Data Processing & Storage: Pandas, SQLite (Structured SQL database layer)

Machine Learning: Scikit-Learn (Random Forest & Logistic Regression architectures)

Data Visualization: Matplotlib, Seaborn (Exploratory charts & Confusion Matrices)

Frontend User Interface: Streamlit (Interactive local web application)
