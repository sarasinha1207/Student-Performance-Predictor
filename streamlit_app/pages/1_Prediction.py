import streamlit as st
import pandas as pd
import numpy as np
import os
import shap
import plotly.graph_objects as go
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import load_object
from src.database import save_prediction, init_db
from src.pdf_generator import generate_pdf_report

st.set_page_config(page_title="Make Prediction", page_icon="🔮", layout="wide")

# Initialize SQLite database
init_db()

# Custom Styling
st.markdown("""
    <style>
    .prediction-title {
        color: #1e3c72;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        font-family: 'Inter', sans-serif;
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
    .recommendation-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2980b9;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 class='prediction-title'>🔮 Student Performance Predictor</h2>", unsafe_allow_html=True)

# Form Layout
with st.form("student_form"):
    st.subheader("📋 Enter Student Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        student_name = st.text_input("Student Name", "Jane Doe")
        previous_score = st.slider("Previous Exam Score (%)", 40.0, 100.0, 75.0, 1.0)
        attendance = st.slider("Attendance Rate (%)", 50.0, 100.0, 90.0, 1.0)
        assignments_completed = st.slider("Assignments Completed", 0, 10, 8, 1)
        
    with col2:
        study_hours = st.slider("Study Hours per Day", 1.0, 10.0, 5.0, 0.5)
        class_participation = st.selectbox("Class Participation Level (1-5)", [1, 2, 3, 4, 5], index=2)
        sleep_hours = st.slider("Sleep Hours per Day", 4.0, 10.0, 7.5, 0.5)
        screen_time = st.slider("Screen Time per Day (hrs)", 1.0, 10.0, 3.0, 0.5)
        
    with col3:
        physical_activity = st.slider("Physical Activity Rate (0-5)", 0, 5, 3, 1)
        parent_education = st.selectbox("Parent Education Level", ["High School", "Some College", "Associate's Degree", "Bachelor's Degree", "Master's Degree"], index=3)
        family_income = st.selectbox("Family Income Status", ["Low", "Medium", "High"], index=1)
        internet_access = st.radio("Internet Access at Home", ["Yes", "No"], horizontal=True)
        school_type = st.radio("School Type", ["Public", "Private"], horizontal=True)
        
    submit_btn = st.form_submit_button("Predict Final Score & Generate Analytics")

# Processing Prediction
if submit_btn:
    if not student_name.strip():
        st.error("Please enter a valid student name.")
    else:
        # Load Pipeline Components
        predict_pipeline = PredictPipeline()
        
        # Structure Input Data
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
        
        # Predict
        prediction = predict_pipeline.predict(input_df)
        predicted_score = float(np.round(prediction[0], 1))
        
        # Determine Grade
        if predicted_score >= 85:
            grade = "A"
        elif predicted_score >= 75:
            grade = "B"
        elif predicted_score >= 60:
            grade = "C"
        elif predicted_score >= 50:
            grade = "D"
        else:
            grade = "F"
            
        # Determine Risk Category
        if predicted_score < 60:
            risk_category = "High Risk"
            risk_class = "risk-high"
        elif predicted_score <= 75:
            risk_category = "Medium Risk"
            risk_class = "risk-medium"
        else:
            risk_category = "Low Risk"
            risk_class = "risk-low"
            
        # Display Results
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.subheader("🔍 Prediction Results")
            st.markdown(f"""
                <div class="result-card {risk_class}">
                    <h3>{student_name}'s Analysis</h3>
                    <h1 style="font-size: 3.5rem; margin: 10px 0; color: white;">{predicted_score}%</h1>
                    <p style="font-size: 1.3rem;">Expected Grade: <b>{grade}</b> | Risk level: <b>{risk_category}</b></p>
                </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.subheader("💡 Dynamic AI Recommendations")
            recs = []
            
            if attendance < 85:
                recs.append("🚨 Improve attendance above 85%. Missing classroom lectures heavily impacts score prediction.")
            if study_hours < 4:
                recs.append("📚 Increase study hours by 1-2 hours daily to build consistency.")
            if screen_time > 5:
                recs.append("📵 Limit excessive screen time (currently above 5 hours) to free up focused study windows.")
            if sleep_hours < 7:
                recs.append("😴 Maintain 7-8 hours of sleep per night to support focus, memory consolidation, and general wellness.")
            if assignments_completed < 8:
                recs.append("📝 Work to complete at least 80% of homework/assignments to ensure concept comprehension.")
            if class_participation < 3:
                recs.append("🙋 Actively engage in classroom discussions; ask questions to clear doubts.")
                
            if not recs:
                recs.append("🌟 Excellent profile! Keep maintaining your outstanding habits to ensure high academic success.")
                
            st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
            for r in recs:
                st.write(r)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Save prediction history to database
        save_prediction(
            student_name=student_name,
            predicted_score=predicted_score,
            grade=grade,
            risk_category=risk_category,
            model_used="Best Selected Model",
            study_hours=study_hours,
            attendance=attendance,
            previous_score=previous_score
        )
        
        # 3. PDF Generator
        inputs = {
            "previous_score": previous_score,
            "attendance": attendance,
            "assignments_completed": assignments_completed,
            "study_hours": study_hours,
            "class_participation": class_participation,
            "sleep_hours": sleep_hours,
            "screen_time": screen_time,
            "physical_activity": physical_activity,
            "parent_education": parent_education,
            "family_income": family_income,
            "internet_access": internet_access,
            "school_type": school_type
        }
        
        pdf_path = generate_pdf_report(
            student_name=student_name,
            inputs=inputs,
            predicted_score=predicted_score,
            grade=grade,
            risk_category=risk_category,
            recommendations=recs
        )
        
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            
        st.download_button(
            label="📄 Download Academic Performance PDF Report",
            data=pdf_bytes,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf",
            use_container_width=True
        )
        
        # 4. SHAP Explainability Graph
        st.subheader("🧠 Explainable AI: SHAP Prediction Factor Breakdown")
        st.write("This interactive chart shows which factors increased (+) or decreased (-) the student's predicted exam score relative to the baseline student average.")
        
        try:
            # Load models
            model_path = os.path.join("models", "model.joblib")
            preprocessor_path = os.path.join("models", "preprocessor.joblib")
            bg_path = os.path.join("models", "background_data.joblib")
            
            if os.path.exists(model_path) and os.path.exists(preprocessor_path) and os.path.exists(bg_path):
                model = load_object(model_path)
                preprocessor = load_object(preprocessor_path)
                background_data = load_object(bg_path)
                
                # Transform inputs
                transformed_input = preprocessor.transform(input_df)
                
                # SHAP Explanation
                explainer = shap.Explainer(model, background_data)
                shap_values = explainer(transformed_input)
                
                # Get transformed feature names
                feature_names = preprocessor.get_feature_names_out()
                contributions = shap_values.values[0]
                
                # Group OneHotEncoded features back to original feature names
                original_features = [
                    "previous_score", "attendance", "assignments_completed", "study_hours",
                    "class_participation", "sleep_hours", "screen_time", "physical_activity",
                    "parent_education", "family_income", "internet_access", "school_type"
                ]
                
                # Map to human-readable names
                readable_names = {
                    "previous_score": "Previous Score",
                    "attendance": "Attendance Rate",
                    "assignments_completed": "Assignments Completed",
                    "study_hours": "Study Hours",
                    "class_participation": "Class Participation",
                    "sleep_hours": "Sleep Duration",
                    "screen_time": "Screen Time",
                    "physical_activity": "Physical Activity",
                    "parent_education": "Parental Education",
                    "family_income": "Family Income",
                    "internet_access": "Internet Access",
                    "school_type": "School Type"
                }
                
                grouped_contributions = {}
                for name, val in zip(feature_names, contributions):
                    parent = None
                    for orig in original_features:
                        if orig in name:
                            parent = orig
                            break
                    if parent:
                        grouped_contributions[readable_names[parent]] = grouped_contributions.get(readable_names[parent], 0.0) + val
                
                # Plotly Visuals
                features_plot = list(grouped_contributions.keys())
                values_plot = list(grouped_contributions.values())
                
                # Sort features by absolute contribution
                sorted_indices = np.argsort(np.abs(values_plot))
                features_plot = [features_plot[idx] for idx in sorted_indices]
                values_plot = [values_plot[idx] for idx in sorted_indices]
                
                # Color code: Green for positive, Red for negative
                colors = ['#27ae60' if val >= 0 else '#c0392b' for val in values_plot]
                
                fig = go.Figure(go.Bar(
                    x=values_plot,
                    y=features_plot,
                    orientation='h',
                    marker_color=colors,
                    text=[f"+{val:.2f}" if val >= 0 else f"{val:.2f}" for val in values_plot],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="SHAP Local Feature Contribution Plot",
                    xaxis_title="Influence on Exam Score (%)",
                    yaxis_title="Student Variables",
                    height=500,
                    margin=dict(l=150, r=50, t=50, b=50),
                    font=dict(family="Inter, sans-serif")
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("Prediction pipeline objects not found. Please train models first.")
        except Exception as ex:
            st.error(f"SHAP chart calculation encountered an error: {str(ex)}")
