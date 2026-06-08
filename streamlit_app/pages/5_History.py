import streamlit as st
import pandas as pd
import os
from src.database import get_prediction_history, clear_prediction_history

st.set_page_config(page_title="Prediction History", page_icon="📜", layout="wide")

st.markdown("""
    <style>
    .header-style {
        color: #1e3c72;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #1e3c72;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 class='header-style'>📜 Prediction Logs & History</h2>", unsafe_allow_html=True)
st.write("Review previous predictions recorded in the SQLite database.")

# Fetch history
history_data = get_prediction_history()

if history_data:
    # Build DataFrame
    columns = [
        "ID", "Student Name", "Timestamp", "Predicted Score (%)", "Expected Grade", 
        "Risk Category", "Model Used", "Study Hours", "Attendance (%)", "Previous Score (%)"
    ]
    df_history = pd.DataFrame(history_data, columns=columns)
    
    # Summary Cards
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
            <div class="metric-card">
                <div>Total Predictions logged</div>
                <div class="metric-value">{len(df_history)}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_m2:
        avg_score = df_history["Predicted Score (%)"].mean()
        st.markdown(f"""
            <div class="metric-card">
                <div>Average Predicted Score</div>
                <div class="metric-value">{avg_score:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_m3:
        high_risk_count = len(df_history[df_history["Risk Category"] == "High Risk"])
        st.markdown(f"""
            <div class="metric-card">
                <div>High Risk Students Logged</div>
                <div class="metric-value" style="color: #c0392b;">{high_risk_count}</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.ln()
    
    # Datatable
    st.dataframe(df_history.drop(columns=["ID"]), use_container_width=True)
    
    # Clear history button
    st.ln()
    if st.button("🗑️ Clear Entire Prediction History Log", use_container_width=True):
        clear_prediction_history()
        st.success("SQLite database prediction history cleared successfully!")
        st.rerun()
        
else:
    st.info("No prediction history logged yet. Run some predictions first!")
