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

# Custom Adaptive Styling (Light/Dark Mode friendly, emoji-free)
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
        padding: 35px;
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
    
    /* KPI Metric styling */
    .kpi-card {
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.14);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 15px;
    }
    .kpi-val {
        font-size: 2rem;
        font-weight: 700;
        color: #3b82f6;
        margin-top: 5px;
    }
    
    /* Leaderboard cards */
    .leaderboard-item {
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.12);
        padding: 12px 20px;
        border-radius: 10px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .rank-badge {
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    .rank-1 { background-color: rgba(245, 158, 11, 0.15); color: #d97706; border: 1px solid rgba(245, 158, 11, 0.3); }
    .rank-2 { background-color: rgba(148, 163, 184, 0.15); color: #64748b; border: 1px solid rgba(148, 163, 184, 0.3); }
    .rank-3 { background-color: rgba(180, 83, 9, 0.15); color: #b45309; border: 1px solid rgba(180, 83, 9, 0.3); }
    .rank-normal { background-color: rgba(128, 128, 128, 0.1); color: gray; }
    
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

# Tabs configuration (Emoji-free)
tab_home, tab_predict, tab_whatif, tab_compare, tab_analytics = st.tabs([
    "Home", "Predict", "What-If Analysis", "Model Comparison", "Analytics"
])

# File paths
data_path = os.path.join("data", "student_data.csv")
model_report_path = os.path.join("models", "model_report.json")

# TAB 1: HOME
with tab_home:
    # 1. Hero Section
    st.markdown("""
        <div style="background-color: rgba(128, 128, 128, 0.05); padding: 30px; border-radius: 14px; border: 1px solid rgba(128, 128, 128, 0.12); margin-bottom: 25px;">
            <h2 style="margin-top: 0; color: #3b82f6;">Proactive Academic Outcomes Advising</h2>
            <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 0;">
                Early prediction of student outcomes enables educational institutions to implement timely support mechanisms. This system evaluates critical academic, lifestyle, and demographic attributes to identify student risk levels and provide tailored strategies for academic growth.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. KPI Cards
    if os.path.exists(data_path):
        df_kpi = pd.read_csv(data_path)
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.markdown(f"""
                <div class="kpi-card">
                    <div style="font-size: 0.9rem; color: gray; font-weight: 500;">Total Students Evaluated</div>
                    <div class="kpi-val">{df_kpi.shape[0]}</div>
                </div>
            """, unsafe_allow_html=True)
        with col_k2:
            st.markdown(f"""
                <div class="kpi-card">
                    <div style="font-size: 0.9rem; color: gray; font-weight: 500;">Average Study Hours/Day</div>
                    <div class="kpi-val">{df_kpi['study_hours'].mean():.1f} hrs</div>
                </div>
            """, unsafe_allow_html=True)
        with col_k3:
            st.markdown(f"""
                <div class="kpi-card">
                    <div style="font-size: 0.9rem; color: gray; font-weight: 500;">Average Class Attendance</div>
                    <div class="kpi-val">{df_kpi['attendance'].mean():.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        with col_k4:
            st.markdown(f"""
                <div class="kpi-card">
                    <div style="font-size: 0.9rem; color: gray; font-weight: 500;">Average Target Final Score</div>
                    <div class="kpi-val">{df_kpi['final_score'].mean():.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
    # 3. Project Summary
    col_ps1, col_ps2 = st.columns(2)
    with col_ps1:
        st.markdown("""
            <div class="adaptive-card">
                <h4>Core Objective</h4>
                <p>The core objective is to build a robust regressor pipeline trained on academic and lifestyle vectors to predict the final exam score of students. This allows teachers to identify students in need of academic support prior to final examinations.</p>
            </div>
        """, unsafe_allow_html=True)
    with col_ps2:
        st.markdown("""
            <div class="adaptive-card">
                <h4>Pipeline Architecture</h4>
                <p>Features are split into numerical columns (scaled using StandardScaler) and categorical columns (encoded using OneHotEncoder) inside a ColumnTransformer. All ML models are evaluated against test partitions using R2, MAE, MSE, and RMSE scores.</p>
            </div>
        """, unsafe_allow_html=True)

# TAB 2: PREDICT
with tab_predict:
    col_p_left, col_p_right = st.columns([3, 2])
    
    with col_p_left:
        st.subheader("Student Details Form")
        with st.form("predict_form"):
            student_name = st.text_input("Student Name", "John Doe")
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                previous_score = st.slider("Previous Score (%)", 40.0, 100.0, 75.0, 1.0)
                attendance = st.slider("Attendance Rate (%)", 50.0, 100.0, 90.0, 1.0)
                assignments_completed = st.slider("Assignments Completed (0-10)", 0, 10, 8, 1)
                study_hours = st.slider("Study Hours/Day", 1.0, 10.0, 5.0, 0.5)
                class_participation = st.selectbox("Class Participation (1-5)", [1, 2, 3, 4, 5], index=3)
                
            with p_col2:
                sleep_hours = st.slider("Sleep Hours/Day", 4.0, 10.0, 7.5, 0.5)
                screen_time = st.slider("Screen Time/Day (hrs)", 1.0, 10.0, 3.0, 0.5)
                physical_activity = st.slider("Physical Activity Rate (0-5)", 0, 5, 3, 1)
                parent_education = st.selectbox("Parent Education Level", ["High School", "Some College", "Associate's Degree", "Bachelor's Degree", "Master's Degree"], index=2)
                family_income = st.selectbox("Family Income Status", ["Low", "Medium", "High"], index=1)
                
            internet_access = st.radio("Internet Access", ["Yes", "No"], horizontal=True)
            school_type = st.radio("School Type", ["Public", "Private"], horizontal=True)
            
            submit = st.form_submit_button("Generate Prediction")
            
    with col_p_right:
        st.subheader("Prediction Analytics")
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
                    
                # 1. Prediction Card
                st.markdown(f"""
                    <div class="result-container {risk_cls}">
                        <h4 style="color: #ffffff; margin: 0;">Predicted Outcome for {student_name}</h4>
                        <div class="result-val">{pred_score}%</div>
                        <p style="font-size: 1.2rem; margin: 0;">Expected Grade: <b>{grade}</b> | Risk Level: <b>{risk_lvl}</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
                # 2. Gauge Chart
                fig_pg = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': "gray"},
                        'bar': {'color': "#2563eb"},
                        'steps': [
                            {'range': [0, 60], 'color': 'rgba(239, 68, 68, 0.15)'},
                            {'range': [60, 75], 'color': 'rgba(245, 158, 11, 0.15)'},
                            {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
                        ]
                    }
                ))
                fig_pg.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': 'gray'})
                st.plotly_chart(fig_pg, use_container_width=True)
                
                # 3. Recommendations
                st.markdown("<h4>AI Recommendations</h4>", unsafe_allow_html=True)
                recs = []
                if attendance < 85:
                    recs.append("Increase attendance above 85% to ensure comprehension of crucial subject topics.")
                if study_hours < 4:
                    recs.append("Dedicate an extra 1-2 hours daily to structured study sessions.")
                if screen_time > 5:
                    recs.append("Minimize non-essential screen time to boost study focus and sleep cycles.")
                if sleep_hours < 7:
                    recs.append("Maintain 7-8 hours of sleep for improved memory consolidation.")
                if assignments_completed < 8:
                    recs.append("Complete pending assignments to strengthen practical knowledge.")
                if not recs:
                    recs.append("Keep maintaining your current positive profile habits.")
                    
                for r in recs:
                    st.write(f"- {r}")
        else:
            st.info("Input student parameters in the form and click Generate Prediction.")

# TAB 3: WHAT-IF ANALYSIS
with tab_whatif:
    col_wi_l, col_wi_r = st.columns([3, 2])
    
    with col_wi_l:
        st.subheader("Adjust Parameters")
        wi_prev = st.slider("Previous Score (%)", 40.0, 100.0, 70.0, 1.0, key="wi_prev_slider")
        wi_att = st.slider("Attendance Rate (%)", 50.0, 100.0, 85.0, 1.0, key="wi_att_slider")
        wi_study = st.slider("Study Hours/Day", 1.0, 10.0, 4.0, 0.5, key="wi_study_slider")
        wi_assign = st.slider("Assignments Completed (0-10)", 0, 10, 6, 1, key="wi_assign_slider")
        wi_sleep = st.slider("Sleep Hours/Day", 4.0, 10.0, 7.0, 0.5, key="wi_sleep_slider")
        wi_screen = st.slider("Screen Time/Day (hrs)", 1.0, 10.0, 5.0, 0.5, key="wi_screen_slider")
        
    with col_wi_r:
        st.subheader("Prediction Change Outcome")
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
            wi_scaled = preprocessor.transform(wi_df)
            wi_pred = model.predict(wi_scaled)
            wi_score = float(np.round(wi_pred[0], 1))
            
            # Baseline Comparison
            baseline_score = 65.0
            delta = wi_score - baseline_score
            delta_txt = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
            
            # Metrics comparison
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Expected Final Score", f"{wi_score}%", delta=delta_txt)
            with col_m2:
                grade_wi = "F"
                if wi_score >= 85: grade_wi = "A"
                elif wi_score >= 75: grade_wi = "B"
                elif wi_score >= 60: grade_wi = "C"
                elif wi_score >= 50: grade_wi = "D"
                st.metric("Expected Grade", grade_wi)
                
            fig_g_wi = go.Figure(go.Indicator(
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
                    ]
                }
            ))
            fig_g_wi.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': 'gray'})
            st.plotly_chart(fig_g_wi, use_container_width=True)
            
        else:
            st.warning("Model files not found. Please train models first.")

# TAB 4: MODEL COMPARISON
with tab_compare:
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
        df_sorted = df_metrics.sort_values(by="R2 Score", ascending=False)
        
        # 1. Model Leaderboard
        st.subheader("Model Leaderboard")
        st.write("Top models ranked by R2 Score:")
        
        ranks = ["rank-1", "rank-2", "rank-3"]
        badges = ["1st Place", "2nd Place", "3rd Place"]
        
        for idx, row in df_sorted.head(3).iterrows():
            st.markdown(f"""
                <div class="leaderboard-item">
                    <div>
                        <span class="rank-badge {ranks[df_sorted.index.get_loc(idx)]}">{badges[df_sorted.index.get_loc(idx)]}</span>
                        <span style="font-weight: bold; margin-left: 15px;">{row['Model']}</span>
                    </div>
                    <div style="font-weight: bold; color: #3b82f6;">R2: {row['R2 Score']:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # 2. Metrics Table
        st.subheader("Evaluation Metrics Table")
        st.dataframe(df_metrics, use_container_width=True)
    else:
        st.warning("Model training report not found. Please run the model trainer pipeline.")

# TAB 5: ANALYTICS
with tab_analytics:
    col_a_l, col_a_r = st.columns([1, 1])
    
    with col_a_l:
        # 1. Feature Importance
        st.subheader("Feature Importance")
        if model is not None and preprocessor is not None:
            # Extract importances/coefficients
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_)
            else:
                importances = np.ones(len(preprocessor.get_feature_names_out()))
                
            # Map back to parent variables
            feature_names = preprocessor.get_feature_names_out()
            original_features = [
                "previous_score", "attendance", "assignments_completed", "study_hours",
                "class_participation", "sleep_hours", "screen_time", "physical_activity",
                "parent_education", "family_income", "internet_access", "school_type"
            ]
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
            
            grouped_importances = {}
            for name, val in zip(feature_names, importances):
                parent = None
                for orig in original_features:
                    if orig in name:
                        parent = orig
                        break
                if parent:
                    readable = readable_names[parent]
                    grouped_importances[readable] = grouped_importances.get(readable, 0.0) + val
            
            total_imp = sum(grouped_importances.values())
            if total_imp > 0:
                for k in grouped_importances:
                    grouped_importances[k] = (grouped_importances[k] / total_imp) * 100
                    
            df_imp = pd.DataFrame({
                "Feature": list(grouped_importances.keys()),
                "Relative Importance (%)": list(grouped_importances.values())
            }).sort_values(by="Relative Importance (%)", ascending=True)
            
            fig_imp = px.bar(
                df_imp,
                x="Relative Importance (%)",
                y="Feature",
                orientation="h",
                color_discrete_sequence=["#3b82f6"]
            )
            fig_imp.update_layout(
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': 'gray'}
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning("Model files not found. Feature Importance cannot be loaded.")
            
        # 3. Correlation Heatmap
        st.subheader("Correlation Heatmap")
        if os.path.exists(data_path):
            df_heat = pd.read_csv(data_path)
            numeric_cols = df_heat.select_dtypes(include=[np.number])
            corr = numeric_cols.corr()
            
            fig_heat = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                labels=dict(color="Correlation")
            )
            fig_heat.update_layout(
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': 'gray'}
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.warning("Dataset not found.")
            
    with col_a_r:
        # 2. Scatter Plot
        st.subheader("Scatter Plot Explorations")
        if os.path.exists(data_path):
            df_scat = pd.read_csv(data_path)
            
            x_option = st.selectbox("Select X-Axis Feature", ["study_hours", "attendance", "previous_score", "sleep_hours", "screen_time"])
            y_option = st.selectbox("Select Y-Axis Feature", ["final_score", "previous_score", "attendance"])
            color_option = st.selectbox("Select Color Legend Feature", ["family_income", "parent_education", "internet_access", "school_type"])
            
            fig_sc = px.scatter(
                df_scat,
                x=x_option,
                y=y_option,
                color=color_option,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_sc.update_layout(
                height=450,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': 'gray'}
            )
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.warning("Dataset not found.")
