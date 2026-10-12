"""
app.py
------
Modern Clean SaaS Dashboard for the Student Performance Predictor.

Author: Ayush Panda
Institution: SRM Institute of Science and Technology, Chennai
"""

import os
import json
import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------
# Page Configuration (Full Screen / Wide Mode)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Clean Light SaaS Dashboard Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    .stApp {
        background-color: #f5f7fa;
        color: #111827;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .block-container {
        padding-top: 0rem;
        padding-bottom: 2.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1400px;
    }

    /* Dark Navy Top Banner */
    .navy-header {
        background-color: #141e28;
        color: #ffffff;
        padding: 2.2rem 2.8rem 4.8rem 2.8rem;
        border-radius: 0 0 20px 20px;
        margin-bottom: -3.2rem;
    }
    .navy-header h1 {
        font-size: 1.9rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.3rem 0;
    }
    .navy-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Overlapping White KPI Banner */
    .kpi-banner {
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 12px 36px rgba(15, 23, 42, 0.07);
        padding: 1.5rem 2rem;
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin-bottom: 1.8rem;
        border: 1px solid #eef2f6;
        position: relative;
        z-index: 10;
    }
    .kpi-col {
        border-right: 1px solid #e2e8f0;
        padding-right: 1rem;
    }
    .kpi-col:last-child {
        border-right: none;
    }
    .kpi-label {
        font-size: 0.84rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.4rem;
    }
    .kpi-val {
        font-size: 2rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.1;
        margin-bottom: 0.35rem;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #94a3b8;
    }
    .up-green { color: #16a34a; font-weight: 700; }
    .down-red { color: #dc2626; font-weight: 700; }

    /* White Bento Card */
    .white-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.6rem;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.045);
        border: 1px solid #eef2f6;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
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
    """Load the serialized Linear Regression model, with automatic cloud fallback."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    data_path = os.path.join(BASE_DIR, "data", "student_performance.csv")
    if os.path.exists(data_path):
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(data_path)
        features = ["study_hours_per_day", "attendance_percent", "assignment_score", "sleep_hours"]
        X = df[features]
        y = df["final_score"]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=42)
        fallback_model = LinearRegression()
        fallback_model.fit(X_train, y_train)
        return fallback_model

    return None


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as file:
            return json.load(file)
    return None


model = load_trained_model()
metrics = load_metrics()

# ---------------------------------------------------------
# Dark Navy Header & Overlapping 4-Column KPI Banner
# ---------------------------------------------------------
r2_val = metrics.get("r2_score", 0.8743) if metrics else 0.8743
mae_val = metrics.get("mae", 2.3739) if metrics else 2.3739
mse_val = metrics.get("mse", 8.9887) if metrics else 8.9887
ds_size = metrics.get("dataset_size", 1000) if metrics else 1000

st.markdown(
    f"""
    <div class="navy-header">
        <h1>Student Performance Predictor</h1>
        <p>This is your Supervised Machine Learning report &amp; real-time score forecast</p>
    </div>

    <div class="kpi-banner">
        <div class="kpi-col">
            <div class="kpi-label">Model R² Accuracy</div>
            <div class="kpi-val">{r2_val * 100:.2f}%</div>
            <div class="kpi-sub"><span class="up-green">&uarr; {r2_val:.4f}</span> variance explained</div>
        </div>
        <div class="kpi-col">
            <div class="kpi-label">Mean Absolute Error</div>
            <div class="kpi-val">{mae_val:.2f}</div>
            <div class="kpi-sub"><span class="up-green">&uarr; &plusmn;{mae_val:.2f} pts</span> avg deviation</div>
        </div>
        <div class="kpi-col">
            <div class="kpi-label">Mean Squared Error</div>
            <div class="kpi-val">{mse_val:.2f}</div>
            <div class="kpi-sub"><span class="down-red">&darr; 2.99 RMSE</span> on test split</div>
        </div>
        <div class="kpi-col">
            <div class="kpi-label">Training &amp; Test Dataset</div>
            <div class="kpi-val">{ds_size:,}</div>
            <div class="kpi-sub"><span class="up-green">&uarr; 800 train</span> / 200 test records</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if model is None:
    st.error("Model file not found! Please run `python train_model.py` first.")
    st.stop()

# ---------------------------------------------------------
# 3-Column Bento Grid (Inputs | Donut Score | Trajectory)
# ---------------------------------------------------------
col1, col2, col3 = st.columns([1.05, 0.95, 1.3], gap="medium")

with col1:
    with st.container(border=True):
        st.markdown("#### Student Factors")
        study_hours = st.slider("🟢 Study Hours / Day", 0.0, 14.0, 6.0, 0.5)
        attendance_percent = st.slider("🔵 Attendance Percentage (%)", 0.0, 100.0, 85.0, 1.0)
        assignment_score = st.slider("🟣 Assignment / Test Score", 0.0, 100.0, 78.0, 1.0)
        sleep_hours = st.slider("🟡 Sleep Hours / Day", 0.0, 14.0, 7.0, 0.5)
        st.button("Predict Score", type="primary", use_container_width=True)

input_data = pd.DataFrame([{
    "study_hours_per_day": float(study_hours),
    "attendance_percent": float(attendance_percent),
    "assignment_score": float(assignment_score),
    "sleep_hours": float(sleep_hours)
}])

raw_prediction = model.predict(input_data)[0]
predicted_score = round(max(0.0, min(100.0, raw_prediction)), 2)

c_study = round(study_hours * 2.9873, 1)
c_att = round(attendance_percent * 0.2407, 1)
c_asgn = round(assignment_score * 0.3771, 1)
c_sleep = round(sleep_hours * 1.2546 + 7.3621, 1)

if predicted_score >= 85.0:
    category = "Excellent"
    badge_bg = "#dcfce7"
    badge_fg = "#15803d"
    advice = "Outstanding academic trajectory! Maintaining these study and attendance habits will yield top-tier results."
elif predicted_score >= 60.0:
    category = "Good"
    badge_bg = "#dbeafe"
    badge_fg = "#1d4ed8"
    advice = "Solid academic standing. Increasing daily study time or assignment preparation can push this score into the 85+ bracket."
else:
    category = "Needs Improvement"
    badge_bg = "#fef3c7"
    badge_fg = "#b45309"
    advice = "Additional academic support recommended. Boosting class attendance and daily study hours will significantly raise performance."

with col2:
    with st.container(border=True):
        st.markdown("#### Predicted Score")
        dash = round((predicted_score / 100.0) * 289, 1)
        st.markdown(
            f"""
            <div style="display:flex;flex-direction:column;align-items:center;margin:0.8rem 0;">
                <div style="position:relative;width:175px;height:175px;display:flex;align-items:center;justify-content:center;">
                    <svg viewBox="0 0 120 120" style="width:100%;height:100%;transform:rotate(-90deg);">
                        <circle cx="60" cy="60" r="46" fill="none" stroke="#f1f5f9" stroke-width="18"></circle>
                        <circle cx="60" cy="60" r="46" fill="none" stroke="#199a59" stroke-width="18"
                                stroke-dasharray="{dash} 289" stroke-linecap="round"></circle>
                    </svg>
                    <div style="position:absolute;text-align:center;">
                        <div style="font-size:1.75rem;font-weight:800;color:#111827;">{predicted_score}</div>
                        <div style="font-size:0.72rem;color:#94a3b8;font-weight:600;">Out of 100</div>
                        <span style="display:inline-block;margin-top:4px;padding:2px 10px;border-radius:99px;font-size:0.72rem;font-weight:700;background:{badge_bg};color:{badge_fg};">{category}</span>
                    </div>
                </div>
            </div>
            <div style="font-size:0.85rem;color:#475569;display:flex;flex-direction:column;gap:8px;margin-top:10px;">
                <div style="display:flex;justify-content:space-between;"><span>🟣 Assignments</span><strong>+{c_asgn} pts</strong></div>
                <div style="display:flex;justify-content:space-between;"><span>🔵 Attendance</span><strong>+{c_att} pts</strong></div>
                <div style="display:flex;justify-content:space-between;"><span>🟢 Study Hours</span><strong>+{c_study} pts</strong></div>
                <div style="display:flex;justify-content:space-between;"><span>🟡 Sleep &amp; Base</span><strong>+{c_sleep} pts</strong></div>
            </div>
            """,
            unsafe_allow_html=True
        )

with col3:
    with st.container(border=True):
        st.markdown("#### Score Trajectory vs. Study Hours")
        x_pos = 50 + (study_hours / 14.0) * 445
        y_pos = max(25, min(165, 165 - ((predicted_score - 45) / 55.0) * 140))
        st.markdown(
            f"""
            <div style="font-size:0.8rem;font-weight:600;color:#475569;margin-bottom:8px;">
                <span style="color:#16a34a;">● Current Student Trajectory</span> &nbsp;&nbsp;
                <span style="color:#eab308;">● Class Average Baseline</span>
            </div>
            <svg viewBox="0 0 520 210" width="100%" height="215">
                <line x1="45" y1="25" x2="500" y2="25" stroke="#f1f5f9" stroke-width="1.5" />
                <line x1="45" y1="70" x2="500" y2="70" stroke="#f1f5f9" stroke-width="1.5" />
                <line x1="45" y1="115" x2="500" y2="115" stroke="#f1f5f9" stroke-width="1.5" />
                <line x1="45" y1="160" x2="500" y2="160" stroke="#f1f5f9" stroke-width="1.5" />
                <text x="10" y="30" fill="#94a3b8" font-size="11" font-weight="600">100</text>
                <text x="14" y="75" fill="#94a3b8" font-size="11" font-weight="600">85</text>
                <text x="14" y="120" fill="#94a3b8" font-size="11" font-weight="600">70</text>
                <text x="14" y="165" fill="#94a3b8" font-size="11" font-weight="600">55</text>
                <rect x="{x_pos - 18:.1f}" y="15" width="36" height="160" rx="8" fill="#f8fafc" />
                <path d="M 50 142 C 160 120, 320 82, 495 45" fill="none" stroke="#eab308" stroke-width="2.5" />
                <path d="M 50 135 C 160 108, 320 68, 495 25" fill="none" stroke="#16a34a" stroke-width="3" />
                <circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="6" fill="#111827" stroke="#ffffff" stroke-width="2.5" />
                <text x="45" y="195" fill="#64748b" font-size="11" font-weight="600">1 hr</text>
                <text x="195" y="195" fill="#64748b" font-size="11" font-weight="600">5 hrs</text>
                <text x="345" y="195" fill="#64748b" font-size="11" font-weight="600">9 hrs</text>
                <text x="475" y="195" fill="#64748b" font-size="11" font-weight="600">13h</text>
            </svg>
            <div style="background:#f8fafc;border:1px solid #eef2f6;border-radius:10px;padding:10px 12px;font-size:0.82rem;color:#475569;margin-top:6px;">
                <strong>Assessment:</strong> {advice}
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# Bottom Card: Visualizations Explorer
# ---------------------------------------------------------
st.markdown("")
with st.container(border=True):
    st.markdown("#### Exploratory Data Analysis & Visualizations")
    v1, v2, v3 = st.columns(3)
    img_study = os.path.join(IMAGES_DIR, "study_hours_vs_score.png")
    img_att = os.path.join(IMAGES_DIR, "attendance_vs_score.png")
    img_asgn = os.path.join(IMAGES_DIR, "assignment_vs_score.png")
    with v1:
        if os.path.exists(img_study):
            st.image(img_study, caption="Study Hours vs Final Score", use_container_width=True)
    with v2:
        if os.path.exists(img_att):
            st.image(img_att, caption="Attendance Percentage vs Final Score", use_container_width=True)
    with v3:
        if os.path.exists(img_asgn):
            st.image(img_asgn, caption="Assignment Score vs Final Score", use_container_width=True)

st.caption(
    "Disclaimer: This prediction is for educational purposes only and should not be treated as an official academic assessment. | Developed by Ayush Panda (SRM Institute of Science and Technology, Chennai)"
)
