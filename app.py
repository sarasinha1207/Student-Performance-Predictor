import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from src.pipeline.predict_pipeline import CustomData
from src.utils import load_object

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Cache model and preprocessor loading for instant prediction response
@st.cache_resource
def load_ml_components():
    model_path = os.path.join("models", "model.joblib")
    preprocessor_path = os.path.join("models", "preprocessor.joblib")
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)
        return model, preprocessor
    return None, None

model, preprocessor = load_ml_components()

# Custom Adaptive Styling (Light/Dark Mode friendly)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
    
    /* Font family overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }
    
    /* Header Banner - Solid Dark Slate Gradient (Contrast safe for both modes) */
    .title-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 30px;
        border-radius: 16px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.05);
    }
    .title-banner h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
    }
    .title-banner p {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-top: 8px;
        margin-bottom: 0;
    }
    
    /* Adaptive Glassmorphic Card Container (Works in Light & Dark Mode) */
    .adaptive-card {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.16);
        padding: 24px;
        border-radius: 14px;
        margin-bottom: 20px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .adaptive-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        border-color: rgba(128, 128, 128, 0.25);
    }
    .adaptive-card h3, .adaptive-card h4 {
        margin-top: 0;
    }
    
    /* Prediction Output Cards with vibrant status-color gradients */
    .result-container {
        padding: 24px;
        border-radius: 14px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .res-low {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    }
    .res-medium {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
    }
    .res-high {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
    }
    .result-val {
        font-size: 3.5rem;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        margin: 8px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-banner">
        <h1>Student Performance Predictor</h1>
        <p>Analyze and predict student exam outcomes using Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# Tabs Configuration (Emoji-free)
tab_about, tab_predict, tab_whatif, tab_compare, tab_analytics = st.tabs([
    "About Project", "Predict Performance", "What-If Analysis", "Model Comparison", "Analytics Dashboard"
])

# File paths
data_path = os.path.join("data", "student_data.csv")
model_report_path = os.path.join("models", "model_report.json")

# TAB 1: ABOUT
with tab_about:
    st.subheader("Project Overview")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
            This application predicts a student's final academic performance based on demographic, lifestyle, and academic features. 
            By analyzing key inputs such as study hours, class attendance, and sleep duration, educators and students can gain actionable insights to improve learning outcomes.
            
            ### Key Predictor Features:
            * **Academic Factors**: Previous Score, Attendance Percentage, Assignments Completed, Study Hours, Class Participation.
            * **Lifestyle Factors**: Sleep Duration, Screen Time, Physical Activity.
            * **Demographic Factors**: Parent Education Level, Family Income Status, Internet Access, School Type.
        """)
    with col2:
        st.markdown("""
            <div class="adaptive-card">
                <h4 style="margin-top: 0;">System Target</h4>
                <p>The target variable is the <b>Final Exam Score (0-100)</b>. The system classifies student performance into risk tiers to provide guidance before actual exams.</p>
            </div>
        """, unsafe_allow_html=True)
        
    if os.path.exists(data_path):
        df_sample = pd.read_csv(data_path)
        st.subheader("Dataset Reference Sample")
        st.dataframe(df_sample.head(6), use_container_width=True)

# TAB 2: PREDICT PERFORMANCE
with tab_predict:
    st.subheader("Input Student Details")
    
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            student_name = st.text_input("Student Name", "John Doe")
            previous_score = st.slider("Previous Score (%)", 40.0, 100.0, 75.0, 1.0)
            attendance = st.slider("Attendance Rate (%)", 50.0, 100.0, 90.0, 1.0)
            assignments_completed = st.slider("Assignments Completed (0-10)", 0, 10, 8, 1)
            
        with col2:
            study_hours = st.slider("Study Hours/Day", 1.0, 10.0, 5.0, 0.5)
            class_participation = st.selectbox("Class Participation (1-5)", [1, 2, 3, 4, 5], index=3)
            sleep_hours = st.slider("Sleep Hours/Day", 4.0, 10.0, 7.5, 0.5)
            screen_time = st.slider("Screen Time/Day (hrs)", 1.0, 10.0, 3.0, 0.5)
            
        with col3:
            physical_activity = st.slider("Physical Activity Rate (0-5)", 0, 5, 3, 1)
            parent_education = st.selectbox("Parent Education Level", ["High School", "Some College", "Associate's Degree", "Bachelor's Degree", "Master's Degree"], index=2)
            family_income = st.selectbox("Family Income Status", ["Low", "Medium", "High"], index=1)
            internet_access = st.radio("Internet Access", ["Yes", "No"], horizontal=True)
            school_type = st.radio("School Type", ["Public", "Private"], horizontal=True)
            
        submit = st.form_submit_button("Predict Score")
        
    if submit:
        if model is None or preprocessor is None:
            st.error("Model files not found. Please train models first.")
        else:
            custom_data = CustomData(
                previous_score=previous_score,
                attendance=attendance,
                assignments_completed=assignments_completed,
                study_hours=study_hours,
                class_participation=class_participation,
                sleep_hours=sleep_hours,
                screen_time=screen_time,
                physical_activity=physical_activity,
                parent_education=parent_education,
                family_income=family_income,
                internet_access=internet_access,
                school_type=school_type
            )
            
            input_df = custom_data.get_data_as_data_frame()
            
            # Predict (extremely fast because model is cached in memory)
            data_scaled = preprocessor.transform(input_df)
            prediction = model.predict(data_scaled)
            pred_score = float(np.round(prediction[0], 1))
            
            # Grade
            if pred_score >= 85: grade = "A"
            elif pred_score >= 75: grade = "B"
            elif pred_score >= 60: grade = "C"
            elif pred_score >= 50: grade = "D"
            else: grade = "F"
            
            # Risk
            if pred_score < 60:
                risk_lvl, risk_cls = "High Risk", "res-high"
            elif pred_score <= 75:
                risk_lvl, risk_cls = "Medium Risk", "res-medium"
            else:
                risk_lvl, risk_cls = "Low Risk", "res-low"
                
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(f"""
                    <div class="result-container {risk_cls}">
                        <h3 style="color: #ffffff; margin: 0;">Prediction for {student_name}</h3>
                        <div class="result-val">{pred_score}%</div>
                        <p style="font-size: 1.25rem; margin: 0;">Expected Grade: <b>{grade}</b> | Risk level: <b>{risk_lvl}</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.markdown("<h4>AI Recommendation:</h4>", unsafe_allow_html=True)
                recs = []
                if attendance < 85:
                    recs.append("Improve attendance rate above 85% to ensure key concepts are covered.")
                if study_hours < 4:
                    recs.append("Increase daily study hours by 1-2 hours.")
                if screen_time > 5:
                    recs.append("Limit non-academic screen time.")
                if sleep_hours < 7:
                    recs.append("Ensure 7-8 hours of sleep per night to maintain focus.")
                if not recs:
                    recs.append("Great profile! Maintain your current academic and lifestyle habits.")
                    
                for r in recs:
                    st.write(f"- {r}")

# TAB 3: WHAT-IF ANALYSIS
with tab_whatif:
    st.subheader("Real-Time Adjustments (What-If)")
    
    col_wi1, col_wi2 = st.columns([3, 2])
    with col_wi1:
        wi_prev = st.slider("Previous Score (%)", 40.0, 100.0, 70.0, 1.0, key="wi_prev_val")
        wi_att = st.slider("Attendance (%)", 50.0, 100.0, 85.0, 1.0, key="wi_att_val")
        wi_study = st.slider("Study Hours/Day", 1.0, 10.0, 4.0, 0.5, key="wi_study_val")
        wi_assign = st.slider("Assignments Completed", 0, 10, 6, 1, key="wi_assign_val")
        wi_sleep = st.slider("Sleep Hours/Day", 4.0, 10.0, 7.0, 0.5, key="wi_sleep_val")
        wi_screen = st.slider("Screen Time/Day (hrs)", 1.0, 10.0, 5.0, 0.5, key="wi_screen_val")
        
    with col_wi2:
        if model is not None and preprocessor is not None:
            wi_custom = CustomData(
                previous_score=wi_prev,
                attendance=wi_att,
                assignments_completed=wi_assign,
                study_hours=wi_study,
                class_participation=3,
                sleep_hours=wi_sleep,
                screen_time=wi_screen,
                physical_activity=3,
                parent_education="Some College",
                family_income="Medium",
                internet_access="Yes",
                school_type="Public"
            )
            wi_df = wi_custom.get_data_as_data_frame()
            
            # Predict
            wi_scaled = preprocessor.transform(wi_df)
            wi_pred = model.predict(wi_scaled)
            wi_score = float(np.round(wi_pred[0], 1))
            
            # Plot Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=wi_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "gray"},
                    'bar': {'color': "#2563eb"},
                    'steps': [
                        {'range': [0, 60], 'color': 'rgba(239, 68, 68, 0.15)'},
                        {'range': [60, 75], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': "#2563eb", 'width': 4},
                        'value': wi_score
                    }
                }
            ))
            
            # Layout color settings adapted to make sure it looks good in light/dark mode
            fig_g.update_layout(
                height=300, 
                margin=dict(l=30, r=30, t=30, b=30),
                paper_bgcolor="rgba(0,0,0,0)", 
                font={'color': 'gray'}
            )
            st.plotly_chart(fig_g, use_container_width=True)
            
            st.write(f"Adjusting variables in real-time updates expected Final Score to **{wi_score}%**.")
        else:
            st.warning("Model files not found. Please train models first.")

# TAB 4: MODEL COMPARISON
with tab_compare:
    st.subheader("Machine Learning Regressor Models Comparison")
    
    if os.path.exists(model_report_path):
        with open(model_report_path, "r") as f:
            report = json.load(f)
            
        records = []
        best_model = ""
        best_r2 = -1.0
        for name, metrics in report.items():
            records.append({
                "Model": name,
                "R2 Score": round(metrics["r2_score"], 4),
                "MAE": round(metrics["mae"], 4),
                "MSE": round(metrics["mse"], 4),
                "RMSE": round(metrics["rmse"], 4)
            })
            if metrics["r2_score"] > best_r2:
                best_r2 = metrics["r2_score"]
                best_model = name
                
        df_metrics = pd.DataFrame(records)
        
        st.markdown(f"""
            <div class="adaptive-card" style="border-left: 5px solid #2563eb;">
                <b>Best Performing Model:</b> {best_model} with an R² Score of <b>{best_r2:.4f}</b>
            </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(df_metrics, use_container_width=True)
        
        # Metric Bar Chart
        metric_choice = st.selectbox("Select metric to compare", ["R2 Score", "MAE", "MSE", "RMSE"])
        
        fig_bar = px.bar(
            df_metrics.sort_values(by=metric_choice, ascending=(metric_choice != "R2 Score")),
            x="Model",
            y=metric_choice,
            color="Model",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_bar.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': 'gray'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("Model training report not found. Please run the model trainer pipeline.")

# TAB 5: ANALYTICS DASHBOARD
with tab_analytics:
    st.subheader("Interactive Exploratory Analytics")
    if os.path.exists(data_path):
        df_anal = pd.read_csv(data_path)
        
        col_an1, col_an2 = st.columns(2)
        with col_an1:
            st.write("**Attendance vs Final Score Scatter**")
            fig_a1 = px.scatter(df_anal, x="attendance", y="final_score", color="school_type", color_discrete_sequence=px.colors.qualitative.Safe)
            fig_a1.update_layout(
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': 'gray'}
            )
            st.plotly_chart(fig_a1, use_container_width=True)
            
        with col_an2:
            st.write("**Study Hours vs Final Score Scatter**")
            fig_a2 = px.scatter(df_anal, x="study_hours", y="final_score", color="family_income", color_discrete_sequence=px.colors.qualitative.Safe)
            fig_a2.update_layout(
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': 'gray'}
            )
            st.plotly_chart(fig_a2, use_container_width=True)
    else:
        st.warning("Dataset not found.")
