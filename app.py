import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import load_object

# page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Styling
st.markdown("""
    <style>
    .title-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }
    .title-banner h1 {
        color: white !important;
        margin: 0;
        font-size: 2.5rem;
    }
    .card-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .risk-medium {
        background: linear-gradient(135deg, #F39C12 0%, #F1C40F 100%);
    }
    .risk-high {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-banner">
        <h1>Student Performance Predictor</h1>
        <p>Analyze and predict student exam outcomes using Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# Tabs
tab_about, tab_predict, tab_whatif, tab_compare, tab_analytics = st.tabs([
    "🏠 About Project", "🔮 Predict Performance", "⚡ What-If Analysis", "📈 Model Comparison", "📊 Analytics Dashboard"
])

# Loading Dataset helper
data_path = os.path.join("data", "student_data.csv")
model_report_path = os.path.join("models", "model_report.json")

# TAB 1: ABOUT
with tab_about:
    st.subheader("📚 Project Overview")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
            This application predicts a student's final academic performance based on demographic, lifestyle, and academic features. 
            By analyzing key inputs such as study hours, class attendance, and sleep duration, educators and students can gain actionable insights to improve learning outcomes.
            
            ### 🛠️ Key Predictor Features:
            * **Academic**: Previous Score, Attendance Percentage, Assignments Completed, Study Hours, Class Participation.
            * **Lifestyle**: Sleep Duration, Screen Time, Physical Activity.
            * **Demographic**: Parent Education Level, Family Income Status, Internet Access, School Type.
        """)
    with col2:
        st.markdown("""
            <div class="card-box">
                <h4 style="color: #1e3c72; margin-top: 0;">🎯 System Target</h4>
                <p>The target variable is the <b>Final Exam Score (0-100)</b>. The system classifies student performance into risk tiers to provide guidance before actual exams.</p>
            </div>
        """, unsafe_allow_html=True)
        
    if os.path.exists(data_path):
        df_sample = pd.read_csv(data_path)
        st.subheader("📊 Dataset Reference Sample")
        st.dataframe(df_sample.head(6), use_container_width=True)

# TAB 2: PREDICT PERFORMANCE
with tab_predict:
    st.subheader("🔮 Input Student Details")
    
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
        # Load prediction
        if not os.path.exists(os.path.join("models", "model.joblib")):
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
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(input_df)
            pred_score = float(np.round(prediction[0], 1))
            
            # Grade
            if pred_score >= 85: grade = "A"
            elif pred_score >= 75: grade = "B"
            elif pred_score >= 60: grade = "C"
            elif pred_score >= 50: grade = "D"
            else: grade = "F"
            
            # Risk
            if pred_score < 60:
                risk_lvl, risk_cls = "High Risk", "risk-high"
            elif pred_score <= 75:
                risk_lvl, risk_cls = "Medium Risk", "risk-medium"
            else:
                risk_lvl, risk_cls = "Low Risk", "risk-low"
                
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(f"""
                    <div class="result-card {risk_cls}">
                        <h3>Prediction for {student_name}</h3>
                        <h1 style="font-size: 3rem; margin: 5px 0; color: white;">{pred_score}%</h1>
                        <p style="font-size: 1.15rem; margin: 0;">Expected Grade: <b>{grade}</b> | Risk level: <b>{risk_lvl}</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.markdown("<h4>💡 AI Recommendation:</h4>", unsafe_allow_html=True)
                recs = []
                if attendance < 85:
                    recs.append("⚠️ Improve your attendance rate above 85% to ensure you don't fall behind.")
                if study_hours < 4:
                    recs.append("📚 Try to increase your daily study hours by 1-2 hours.")
                if screen_time > 5:
                    recs.append("📵 Limit excessive non-academic screen time.")
                if sleep_hours < 7:
                    recs.append("😴 Ensure 7-8 hours of sleep per night to maintain focus.")
                if not recs:
                    recs.append("✅ Great profile! Maintain your current academic and lifestyle habits.")
                    
                for r in recs:
                    st.write(r)

# TAB 3: WHAT-IF ANALYSIS
with tab_whatif:
    st.subheader("⚡ Real-Time Adjustments (What-If)")
    
    col_wi1, col_wi2 = st.columns([3, 2])
    with col_wi1:
        wi_prev = st.slider("Previous Score (%)", 40.0, 100.0, 70.0, 1.0, key="wi_prev_val")
        wi_att = st.slider("Attendance (%)", 50.0, 100.0, 85.0, 1.0, key="wi_att_val")
        wi_study = st.slider("Study Hours/Day", 1.0, 10.0, 4.0, 0.5, key="wi_study_val")
        wi_assign = st.slider("Assignments Completed", 0, 10, 6, 1, key="wi_assign_val")
        wi_sleep = st.slider("Sleep Hours/Day", 4.0, 10.0, 7.0, 0.5, key="wi_sleep_val")
        wi_screen = st.slider("Screen Time/Day (hrs)", 1.0, 10.0, 5.0, 0.5, key="wi_screen_val")
        
    with col_wi2:
        if os.path.exists(os.path.join("models", "model.joblib")):
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
            wi_pred = PredictPipeline().predict(wi_df)
            wi_score = float(np.round(wi_pred[0], 1))
            
            # Plot Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=wi_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1e3c72"},
                    'steps': [
                        {'range': [0, 60], 'color': '#ffcccc'},
                        {'range': [60, 75], 'color': '#fff2cc'},
                        {'range': [75, 100], 'color': '#ccffdd'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'value': wi_score
                    }
                }
            ))
            fig_g.update_layout(height=300, margin=dict(l=30, r=30, t=30, b=30))
            st.plotly_chart(fig_g, use_container_width=True)
            
            st.write(f"Adjusting variables in real-time updates expected Final Score to **{wi_score}%**.")
        else:
            st.warning("Model files not found. Please train models first.")

# TAB 4: MODEL COMPARISON
with tab_compare:
    st.subheader("📈 Machine Learning Regressor Models Comparison")
    
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
            <div style="background-color: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 5px solid #2980b9; margin-bottom: 20px;">
                💡 <b>Best Performing Model:</b> {best_model} with an R² Score of <b>{best_r2:.4f}</b>
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
            title=f"Models Compared by {metric_choice}"
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("Model training report not found. Please run the model trainer pipeline.")

# TAB 5: ANALYTICS DASHBOARD
with tab_analytics:
    st.subheader("📊 Interactive Exploratory Analytics")
    if os.path.exists(data_path):
        df_anal = pd.read_csv(data_path)
        
        col_an1, col_an2 = st.columns(2)
        with col_an1:
            st.write("**Attendance vs Final Score Scatter**")
            fig_a1 = px.scatter(df_anal, x="attendance", y="final_score", color="school_type")
            fig_a1.update_layout(height=380)
            st.plotly_chart(fig_a1, use_container_width=True)
            
        with col_an2:
            st.write("**Study Hours vs Final Score Scatter**")
            fig_a2 = px.scatter(df_anal, x="study_hours", y="final_score", color="family_income")
            fig_a2.update_layout(height=380)
            st.plotly_chart(fig_a2, use_container_width=True)
    else:
        st.warning("Dataset not found.")
