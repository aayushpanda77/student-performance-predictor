"""
app.py
------
Streamlit Web Application for the Student Performance Predictor.
Allows users to enter study habits, attendance, and lifestyle factors to predict final academic scores.

Author: Ayush Panda
Institution: SRM Institute of Science and Technology, Chennai
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# File Paths & Model Loading
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

@st.cache_resource
def load_trained_model():
    """Load the serialized machine learning model from disk."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def load_metrics():
    """Load model evaluation metrics if available."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

model = load_trained_model()
metrics = load_metrics()

# ---------------------------------------------------------
# Sidebar Information
# ---------------------------------------------------------
with st.sidebar:
    st.header("📌 About the Project")
    st.markdown(
        """
        The **Student Performance Predictor** is a supervised machine learning application 
        built using **Linear Regression** to predict a student's final academic score based 
        on key academic habits and lifestyle metrics.
        
        **Author:**
        - **Ayush Panda**
        - B.Tech Artificial Intelligence
        - SRM Institute of Science and Technology, Chennai
        """
    )
    
    st.divider()
    
    st.subheader("📊 Model Evaluation Metrics")
    if metrics:
        st.markdown(f"- **Algorithm:** `{metrics.get('algorithm', 'Linear Regression')}`")
        st.markdown(f"- **MAE (Mean Absolute Error):** `{metrics.get('mae', 'N/A')}`")
        st.markdown(f"- **MSE (Mean Squared Error):** `{metrics.get('mse', 'N/A')}`")
        st.markdown(f"- **RMSE:** `{metrics.get('rmse', 'N/A')}`")
        st.markdown(f"- **R² Score:** `{metrics.get('r2_score', 'N/A')}` ({metrics.get('r2_score', 0) * 100:.2f}%)")
    else:
        st.info("Metrics will be displayed here once `train_model.py` is executed.")
        
    st.divider()
    st.caption("Built with Python, Scikit-learn, Pandas & Streamlit.")

# ---------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------
st.title("🎓 Student Performance Predictor")
st.write("Predict a student's expected final score using academic and lifestyle factors.")
st.markdown("---")

# Verify model availability
if model is None:
    st.error("⚠️ **Model file not found!**")
    st.warning("Please run `python train_model.py` from the project root to train and save the model before using this application.")
    st.stop()

# ---------------------------------------------------------
# Input Section
# ---------------------------------------------------------
st.subheader("📝 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input(
        label="📖 Study Hours per Day",
        min_value=0.0,
        max_value=14.0,
        value=5.0,
        step=0.5,
        help="Average number of hours dedicated to self-study each day (0.0 to 14.0 hours)."
    )
    
    attendance_percent = st.number_input(
        label="📅 Attendance Percentage (%)",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0,
        help="Class attendance percentage over the semester (0% to 100%)."
    )

with col2:
    assignment_score = st.number_input(
        label="📝 Assignment / Test Score (out of 100)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0,
        help="Average score achieved in assignments, quizzes, and internal tests (0 to 100)."
    )
    
    sleep_hours = st.number_input(
        label="😴 Sleep Hours per Day",
        min_value=0.0,
        max_value=14.0,
        value=7.0,
        step=0.5,
        help="Average hours of sleep per night (0.0 to 14.0 hours)."
    )

st.markdown("")

# ---------------------------------------------------------
# Prediction Logic & Output
# ---------------------------------------------------------
predict_button = st.button("🔮 Predict Score", type="primary", use_container_width=True)

if predict_button:
    # Prepare feature input DataFrame matching training columns
    input_data = pd.DataFrame([{
        'study_hours_per_day': float(study_hours),
        'attendance_percent': float(attendance_percent),
        'assignment_score': float(assignment_score),
        'sleep_hours': float(sleep_hours)
    }])
    
    # Generate prediction using the trained model
    raw_prediction = model.predict(input_data)[0]
    
    # Academic score is realistically bounded between 0 and 100
    clamped_score = max(0.0, min(100.0, raw_prediction))
    predicted_score = round(clamped_score, 2)
    
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    
    # Display Score in structured metric cards
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(
            label="Predicted Score",
            value=f"{predicted_score} / 100"
        )
    
    with res_col2:
        # Determine performance category and interpretation
        if predicted_score >= 85.0:
            category = "Excellent"
            st.success(f"**Interpretation:** {category} 🌟")
            st.write("The student shows strong academic habits and understanding. Consistent practice will help maintain top performance!")
        elif predicted_score >= 60.0:
            category = "Good"
            st.info(f"**Interpretation:** {category} 👍")
            st.write("The student is performing solidly. Improving daily study consistency or test preparation can elevate scores into the top bracket.")
        else:
            category = "Needs Improvement"
            st.warning(f"**Interpretation:** {category} ⚠️")
            st.write("The predicted score suggests additional support is recommended. Increasing regular study hours, attending classes, and seeking teacher feedback can make a big impact.")

    st.markdown("")
    st.info("ℹ️ **Disclaimer:** This prediction is for educational purposes only and should not be treated as an official academic assessment.")

# ---------------------------------------------------------
# Exploratory Data Analysis & Visualizations Section
# ---------------------------------------------------------
st.markdown("---")
with st.expander("📈 Explore Dataset Visualizations & Insights"):
    st.write("These charts illustrate the relationships identified during exploratory data analysis:")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Study Hours", "Attendance", "Assignments", "Correlation"])
    
    with tab1:
        img_study = os.path.join(IMAGES_DIR, "study_hours_vs_score.png")
        if os.path.exists(img_study):
            st.image(img_study, caption="Study Hours per Day vs Final Score", use_container_width=True)
            st.caption("Demonstrates a strong positive linear correlation: more consistent daily study leads to higher exam performance.")
        else:
            st.info("Run `python train_model.py` to generate this chart.")
            
    with tab2:
        img_att = os.path.join(IMAGES_DIR, "attendance_vs_score.png")
        if os.path.exists(img_att):
            st.image(img_att, caption="Attendance Percentage vs Final Score", use_container_width=True)
            st.caption("Demonstrates steady positive relationship: students attending classes regularly consistently outperform those with low attendance.")
        else:
            st.info("Run `python train_model.py` to generate this chart.")
            
    with tab3:
        img_asgn = os.path.join(IMAGES_DIR, "assignment_vs_score.png")
        if os.path.exists(img_asgn):
            st.image(img_asgn, caption="Assignment Score vs Final Score", use_container_width=True)
            st.caption("Demonstrates strong correlation: internal assessments provide a strong indicator of final term comprehension.")
        else:
            st.info("Run `python train_model.py` to generate this chart.")

    with tab4:
        img_corr = os.path.join(IMAGES_DIR, "correlation_heatmap.png")
        if os.path.exists(img_corr):
            st.image(img_corr, caption="Feature Correlation Heatmap", use_container_width=True)
            st.caption("Visualizes Pearson correlation coefficients between all academic and lifestyle variables.")
        else:
            st.info("Run `python train_model.py` to generate this chart.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "Student Performance Predictor | Developed by Ayush Panda (SRM Institute of Science and Technology, Chennai)"
)
