import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white !important;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        margin: 0;
        font-size: 2.8rem;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 10px;
    }
    .card {
        background-color: white;
        padding: 24px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #1e3c72;
        margin-bottom: 20px;
    }
    .card h3 {
        color: #1e3c72;
        margin-top: 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2a5298;
    }
    </style>
""", unsafe_allow_html=True)

# Main Banner
st.markdown("""
    <div class="main-header">
        <h1>Student Performance Predictor</h1>
        <p>A production-ready end-to-end Machine Learning web application designed to predict academic outcomes and detect student risk levels using advanced predictive modeling.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎓 Navigation")
st.sidebar.markdown("""
Welcome to the Student Performance Predictor dashboard. Use the sidebar pages to:
* **🔮 Make Predictions**: Input parameters and generate PDF reports.
* **⚡ What-If Analysis**: Interactively slide inputs to see changes.
* **📈 Model Comparison**: Check performance across 7 ML models.
* **📊 Analytics Dashboard**: Visualize correlation and distributions.
* **📜 Logs History**: View past prediction records.
""")

# Layout Columns
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
        <div class="card">
            <h3>📌 Project Objective & Problem Statement</h3>
            <p>Predicting student academic failure or success early allows educators to deliver timely, personalized interventions. This system acts as a supportive advisory engine for students, teachers, and school administrations by providing:</p>
            <ul>
                <li><b>Early Risk Identification</b>: Classifying students into High, Medium, and Low risk thresholds.</li>
                <li><b>Behavioral Advisory</b>: Generating dynamically tailored recommendations based on lifestyle and academic features.</li>
                <li><b>Explainable AI (XAI)</b>: Informing users on exactly <i>why</i> a score was predicted using SHAP values.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Dataset Overview")
    data_path = os.path.join("data", "student_data.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        # Display dataset statistics
        st.write("Below is a sample of the historical dataset used to train the machine learning models:")
        st.dataframe(df.head(6), use_container_width=True)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
                <div class="metric-box">
                    <div>Total Records</div>
                    <div class="metric-value">{df.shape[0]}</div>
                </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
                <div class="metric-box">
                    <div>Total Features</div>
                    <div class="metric-value">{df.shape[1] - 1}</div>
                </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
                <div class="metric-box">
                    <div>Target Column</div>
                    <div class="metric-value">Final Score</div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Dataset CSV not found. It will be generated automatically when you run the pipeline.")

with col2:
    st.markdown("""
        <div class="card">
            <h3>🔬 Machine Learning Models</h3>
            <p>Our pipeline compares 7 distinct regressor models using cross-validated Grid Search to output the most accurate predictions:</p>
            <ol>
                <li><b>Linear Regression</b></li>
                <li><b>Decision Tree Regressor</b></li>
                <li><b>Random Forest Regressor</b></li>
                <li><b>AdaBoost Regressor</b></li>
                <li><b>Gradient Boosting Regressor</b></li>
                <li><b>XGBoost Regressor</b></li>
                <li><b>CatBoost Regressor</b></li>
            </ol>
            <p>The best model is automatically pickled and served for predictions.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="card" style="border-left-color: #27ae60;">
            <h3>🛠️ Dataset Schema Features</h3>
            <p>Predictions are derived from three distinct vectors:</p>
            <ul>
                <li><b>Academic Vectors</b>: Previous Exam Scores, Attendance Percentage, Assignments Completed, Study Hours, Class Participation.</li>
                <li><b>Lifestyle Vectors</b>: Sleep Hours, Screen Time, Physical Activity.</li>
                <li><b>Demographic Vectors</b>: Parent Education, Family Income, Internet Access, School Type.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
