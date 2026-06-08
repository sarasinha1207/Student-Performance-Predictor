import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .header-style {
        color: #1e3c72;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 class='header-style'>📊 Exploratory Analytics Dashboard</h2>", unsafe_allow_html=True)
st.write("Explore relationship distributions, academic trends, and feature correlations in the student performance dataset.")

data_path = os.path.join("data", "student_data.csv")

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Tabs for different analyses
    tab_corr, tab_scatter, tab_dist = st.tabs(["🔥 Correlation Heatmap", "📈 Scatter Analyses", "📊 Feature Distributions"])
    
    with tab_corr:
        st.subheader("Numerical Correlation Heatmap")
        st.write("Understand which student variables have the strongest linear relationship with the target `final_score`.")
        
        corr_matrix = numeric_df.corr()
        
        fig_heat = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            labels=dict(color="Correlation Coefficient")
        )
        fig_heat.update_layout(
            height=600,
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with tab_scatter:
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.subheader("Attendance vs Final Exam Score")
            fig_scat1 = px.scatter(
                df,
                x="attendance",
                y="final_score",
                color="family_income",
                hover_data=["study_hours", "previous_score"],
                trendline="ols",
                labels={"attendance": "Attendance (%)", "final_score": "Final Score (%)"},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_scat1.update_layout(height=450, font=dict(family="Inter, sans-serif"))
            st.plotly_chart(fig_scat1, use_container_width=True)
            
        with col_s2:
            st.subheader("Study Hours vs Final Exam Score")
            fig_scat2 = px.scatter(
                df,
                x="study_hours",
                y="final_score",
                color="internet_access",
                hover_data=["attendance", "previous_score"],
                trendline="ols",
                labels={"study_hours": "Study Hours / Day", "final_score": "Final Score (%)"},
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_scat2.update_layout(height=450, font=dict(family="Inter, sans-serif"))
            st.plotly_chart(fig_scat2, use_container_width=True)
            
    with tab_dist:
        col_d1, col_d2 = st.columns([1, 2])
        
        with col_d1:
            st.subheader("Select Feature to Plot")
            dist_col = st.selectbox(
                "Numerical Feature",
                ["final_score", "previous_score", "attendance", "study_hours", "sleep_hours", "screen_time"]
            )
            
            st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 15px;">
                    <p style="margin:0;">💡 <b>Distribution insights:</b> A normal distribution indicates balanced variables, while a skewed histogram highlights features with extreme values (e.g., highly clustered attendance rates).</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col_d2:
            fig_dist = px.histogram(
                df,
                x=dist_col,
                kde=True,
                color_discrete_sequence=["#1e3c72"],
                labels={dist_col: dist_col.replace('_', ' ').title()}
            )
            fig_dist.update_layout(
                title=f"Distribution of {dist_col.replace('_', ' ').title()}",
                height=450,
                yaxis_title="Count",
                font=dict(family="Inter, sans-serif")
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
else:
    st.warning("Student dataset file not found. Please train models first.")
