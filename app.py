"""
app.py
------
Modern, full-screen Streamlit web dashboard for the Student Performance Predictor.

Author: Ayush Panda
Institution: SRM Institute of Science and Technology, Chennai
"""

import os
import json
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration (Full Screen / Wide Mode)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Modern Custom Styling (CSS)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Clean page padding for full-screen dashboard feel */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 100%;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        color: #ffffff;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.08rem;
        color: #cbd5e1;
        margin: 0 0 1.2rem 0;
    }
    .badge-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Prediction Result Hero Card */
    .result-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid rgba(129, 140, 248, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
    }
    .score-label {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }
    .score-value {
        font-size: 4.2rem;
        font-weight: 800;
        line-height: 1.1;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
    }
    .status-pill {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    .status-excellent {
        background: rgba(34, 197, 94, 0.18);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.4);
    }
    .status-good {
        background: rgba(56, 189, 248, 0.18);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }
    .status-improve {
        background: rgba(251, 191, 36, 0.18);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.4);
    }

    /* Metric KPI Cards */
    .kpi-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 0.3rem;
    }
    .kpi-num {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f8fafc;
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

    # Fallback: train directly from dataset if cloud Python/sklearn version differs
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
    """Load evaluation metrics from disk."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as file:
            return json.load(file)
    return None


model = load_trained_model()
metrics = load_metrics()

# ---------------------------------------------------------
# Top Hero Banner
# ---------------------------------------------------------
r2_badge = f"R² Score: {metrics['r2_score'] * 100:.1f}%" if metrics else "Supervised ML"
st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-title">🎓 Student Performance Predictor</div>
        <div class="hero-subtitle">
            Predict a student's expected final score using academic and lifestyle factors powered by Supervised Machine Learning.
        </div>
        <div class="badge-row">
            <span class="hero-badge">Algorithm: Linear Regression</span>
            <span class="hero-badge">{r2_badge}</span>
            <span class="hero-badge">1,000 Student Dataset</span>
            <span class="hero-badge">SRM IST AI Project</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if model is None:
    st.error("⚠️ Model file not found! Please run `python train_model.py` first.")
    st.stop()

# ---------------------------------------------------------
# Main Full-Screen Two-Column Workspace
# ---------------------------------------------------------
left_col, right_col = st.columns([1.05, 1.15], gap="large")

with left_col:
    st.subheader("🎛️ Student Profile Inputs")
    with st.container(border=True):
        st.caption("Adjust the sliders or enter values below to simulate student performance:")

        in_col1, in_col2 = st.columns(2)
        with in_col1:
            study_hours = st.slider(
                label="📖 Study Hours per Day",
                min_value=0.0,
                max_value=14.0,
                value=6.0,
                step=0.5,
                help="Average daily self-study hours."
            )
            attendance_percent = st.slider(
                label="📅 Attendance Percentage (%)",
                min_value=0.0,
                max_value=100.0,
                value=85.0,
                step=1.0,
                help="Overall class attendance percentage."
            )

        with in_col2:
            assignment_score = st.slider(
                label="📝 Assignment / Test Score",
                min_value=0.0,
                max_value=100.0,
                value=78.0,
                step=1.0,
                help="Average score across assignments and internal tests."
            )
            sleep_hours = st.slider(
                label="😴 Sleep Hours per Day",
                min_value=0.0,
                max_value=14.0,
                value=7.0,
                step=0.5,
                help="Average nightly rest duration."
            )

        st.markdown("")
        predict_button = st.button("🚀 Predict Final Score", type="primary", use_container_width=True)

# Calculate prediction (runs on button click or defaults to initial profile so dashboard is never empty)
input_data = pd.DataFrame([{
    "study_hours_per_day": float(study_hours),
    "attendance_percent": float(attendance_percent),
    "assignment_score": float(assignment_score),
    "sleep_hours": float(sleep_hours)
}])

raw_prediction = model.predict(input_data)[0]
predicted_score = round(max(0.0, min(100.0, raw_prediction)), 2)

if predicted_score >= 85.0:
    category = "Excellent"
    pill_class = "status-excellent"
    advice = "Outstanding academic trajectory! Maintaining these study and attendance habits will yield top-tier results."
elif predicted_score >= 60.0:
    category = "Good"
    pill_class = "status-good"
    advice = "Solid academic standing. Increasing daily study time or assignment preparation can push this score into the 85+ bracket."
else:
    category = "Needs Improvement"
    pill_class = "status-improve"
    advice = "Academic intervention recommended. Boosting class attendance and dedicating more hours to daily study will significantly raise performance."

with right_col:
    st.subheader("🎯 Prediction & Score Breakdown")

    st.markdown(
        f"""
        <div class="result-card">
            <div class="score-label">Predicted Final Academic Score</div>
            <div class="score-value">{predicted_score} <span style="font-size:1.8rem;color:#64748b;">/ 100</span></div>
            <div class="status-pill {pill_class}">{category}</div>
            <p style="color:#cbd5e1;font-size:0.96rem;margin:0;">{advice}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Score Progress Meter & Factor Contributions
    with st.container(border=True):
        st.caption("**Score Meter (Out of 100)**")
        st.progress(int(round(predicted_score)))

        if metrics and "coefficients" in metrics:
            coefs = metrics["coefficients"]
            b_study = round(study_hours * coefs.get("study_hours_per_day", 0), 1)
            b_att = round(attendance_percent * coefs.get("attendance_percent", 0), 1)
            b_asgn = round(assignment_score * coefs.get("assignment_score", 0), 1)
            b_sleep = round(sleep_hours * coefs.get("sleep_hours", 0), 1)

            st.caption("**Estimated Points Contributed by Each Factor:**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Study Hours", f"+{b_study} pts")
            c2.metric("Attendance", f"+{b_att} pts")
            c3.metric("Assignments", f"+{b_asgn} pts")
            c4.metric("Sleep Rest", f"+{b_sleep} pts")

    st.caption(
        "ℹ️ **Disclaimer:** This prediction is for educational purposes only and should not be treated as an official academic assessment."
    )

# ---------------------------------------------------------
# Model Evaluation Metrics Bar (Full Width)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Model Evaluation & Analytics Dashboard")

if metrics:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-title">R² Accuracy Score</div><div class="kpi-num">{metrics['r2_score']:.4f} ({metrics['r2_score']*100:.1f}%)</div></div>""",
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-title">Mean Absolute Error (MAE)</div><div class="kpi-num">{metrics['mae']:.4f}</div></div>""",
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-title">Mean Squared Error (MSE)</div><div class="kpi-num">{metrics['mse']:.4f}</div></div>""",
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-title">Dataset Records</div><div class="kpi-num">{metrics.get('dataset_size', 1000)}</div></div>""",
            unsafe_allow_html=True
        )

st.markdown("")

# ---------------------------------------------------------
# Interactive Visual Insights Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📈 Feature Relationships",
    "🔥 Correlation & Score Distribution",
    "🎯 Model Accuracy (Actual vs Predicted)"
])

with tab1:
    v_col1, v_col2, v_col3 = st.columns(3)
    img_study = os.path.join(IMAGES_DIR, "study_hours_vs_score.png")
    img_att = os.path.join(IMAGES_DIR, "attendance_vs_score.png")
    img_asgn = os.path.join(IMAGES_DIR, "assignment_vs_score.png")

    with v_col1:
        if os.path.exists(img_study):
            st.image(img_study, caption="Study Hours vs Final Score", use_container_width=True)
    with v_col2:
        if os.path.exists(img_att):
            st.image(img_att, caption="Attendance Percentage vs Final Score", use_container_width=True)
    with v_col3:
        if os.path.exists(img_asgn):
            st.image(img_asgn, caption="Assignment Score vs Final Score", use_container_width=True)

with tab2:
    h_col1, h_col2 = st.columns(2)
    img_corr = os.path.join(IMAGES_DIR, "correlation_heatmap.png")
    img_dist = os.path.join(IMAGES_DIR, "score_distribution.png")

    with h_col1:
        if os.path.exists(img_corr):
            st.image(img_corr, caption="Feature Correlation Heatmap", use_container_width=True)
    with h_col2:
        if os.path.exists(img_dist):
            st.image(img_dist, caption="Distribution of Final Scores", use_container_width=True)

with tab3:
    a_col1, a_col2 = st.columns([1.2, 1])
    img_pred = os.path.join(IMAGES_DIR, "actual_vs_predicted.png")
    with a_col1:
        if os.path.exists(img_pred):
            st.image(img_pred, caption="Actual vs Predicted Final Scores (Test Set)", use_container_width=True)
    with a_col2:
        st.markdown("### How the Linear Regression Model Works")
        st.write(
            "The model learns a linear equation mapping the four student inputs to the expected final score:"
        )
        st.latex(r"\hat{y} = \beta_0 + \beta_1 X_{\text{study}} + \beta_2 X_{\text{attendance}} + \beta_3 X_{\text{assignment}} + \beta_4 X_{\text{sleep}}")
        if metrics and "coefficients" in metrics:
            st.markdown("**Learned Weights:**")
            st.markdown(f"- **Intercept ($\\beta_0$):** `{metrics.get('intercept', 0)}`")
            for feat, val in metrics["coefficients"].items():
                st.markdown(f"- **`{feat}`:** `+{val}`")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "Developed by **Ayush Panda** | B.Tech Artificial Intelligence | SRM Institute of Science and Technology, Chennai"
)
