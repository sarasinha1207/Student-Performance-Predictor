import streamlit as st
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="What-If Analysis", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    .header-style {
        color: #1e3c72;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        margin-bottom: 15px;
    }
    .score-card {
        background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%);
        padding: 30px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 class='header-style'>⚡ What-If Analysis Module</h2>", unsafe_allow_html=True)
st.write("Adjust academic, lifestyle, and demographic parameters using sliders to instantly visualize how changing behaviors impacts the student's predicted Final Score.")

# We want real-time predictions as they move sliders (no submit form button)
col_inputs, col_gauge = st.columns([3, 2])

with col_inputs:
    st.subheader("🛠️ Adjust Student Parameters")
    
    tab_acad, tab_life, tab_demo = st.tabs(["📚 Academic Factors", "🏃 Lifestyle Factors", "🏠 Demographic Factors"])
    
    with tab_acad:
        previous_score = st.slider("Previous Exam Score (%)", 40.0, 100.0, 70.0, 1.0, key="wi_prev")
        attendance = st.slider("Attendance Rate (%)", 50.0, 100.0, 85.0, 1.0, key="wi_att")
        study_hours = st.slider("Study Hours per Day", 1.0, 10.0, 4.0, 0.5, key="wi_study")
        assignments_completed = st.slider("Assignments Completed", 0, 10, 6, 1, key="wi_assign")
        class_participation = st.selectbox("Class Participation Level (1-5)", [1, 2, 3, 4, 5], index=2, key="wi_part")
        
    with tab_life:
        sleep_hours = st.slider("Sleep Hours per Day", 4.0, 10.0, 7.0, 0.5, key="wi_sleep")
        screen_time = st.slider("Screen Time per Day (hrs)", 1.0, 10.0, 5.0, 0.5, key="wi_screen")
        physical_activity = st.slider("Physical Activity Rate (0-5)", 0, 5, 2, 1, key="wi_phys")
        
    with tab_demo:
        parent_education = st.selectbox("Parent Education Level", ["High School", "Some College", "Associate's Degree", "Bachelor's Degree", "Master's Degree"], index=1, key="wi_parent")
        family_income = st.selectbox("Family Income Status", ["Low", "Medium", "High"], index=1, key="wi_income")
        internet_access = st.radio("Internet Access at Home", ["Yes", "No"], horizontal=True, key="wi_net")
        school_type = st.radio("School Type", ["Public", "Private"], horizontal=True, key="wi_school")

# Run real-time prediction
model_path = os.path.join("models", "model.joblib")
if os.path.exists(model_path):
    predict_pipeline = PredictPipeline()
    
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
    prediction = predict_pipeline.predict(input_df)
    predicted_score = float(np.round(prediction[0], 1))
    
    # Baseline comparison (average baseline)
    baseline_score = 65.0 # Let's assume a standard student baseline is 65
    difference = predicted_score - baseline_score
    diff_text = f"+{difference:.1f}%" if difference >= 0 else f"{difference:.1f}%"
    diff_color = "#27ae60" if difference >= 0 else "#c0392b"
    
    with col_gauge:
        st.subheader("🎯 Real-Time Score Prediction")
        
        # Plotly Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = predicted_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Final Exam Score", 'font': {'size': 20, 'family': 'Outfit, sans-serif'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#2a5298"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 60], 'color': '#ffcccc'},
                    {'range': [60, 75], 'color': '#fff2cc'},
                    {'range': [75, 100], 'color': '#ccffdd'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': predicted_score
                }
            }
        ))
        
        fig.update_layout(height=350, margin=dict(l=30, r=30, t=50, b=30))
        st.plotly_chart(fig, use_container_width=True)
        
        # Display dynamic metrics
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(
                label="Predicted Final Score",
                value=f"{predicted_score}%",
                delta=diff_text,
                delta_color="normal"
            )
        with col_m2:
            grade = "F"
            if predicted_score >= 85: grade = "A"
            elif predicted_score >= 75: grade = "B"
            elif predicted_score >= 60: grade = "C"
            elif predicted_score >= 50: grade = "D"
            
            risk_lvl = "High Risk"
            if predicted_score >= 75: risk_lvl = "Low Risk"
            elif predicted_score >= 60: risk_lvl = "Medium Risk"
            
            st.metric(
                label="Expected Grade",
                value=grade,
                delta=risk_lvl,
                delta_color="inverse"
            )
            
        st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #2a5298;">
                <p style="margin: 0; font-size: 0.95rem;">💡 <b>What-If Summary:</b> Adjusting these inputs pushes the student's expected performance to <b>{predicted_score}%</b>, putting them in the <b>{risk_lvl}</b> category.</p>
            </div>
        """, unsafe_allow_html=True)
else:
    st.warning("Model files not found. Please train models first.")
