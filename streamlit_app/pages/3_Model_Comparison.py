import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .header-style {
        color: #1e3c72;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        margin-bottom: 15px;
    }
    .highlight-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 class='header-style'>📈 Machine Learning Model Comparison</h2>", unsafe_allow_html=True)
st.write("View performance metrics across all 7 trained regression models. The models are evaluated on test data using R² Score, MAE, MSE, and RMSE.")

report_path = os.path.join("models", "model_report.json")

if os.path.exists(report_path):
    with open(report_path, "r") as f:
        model_report = json.load(f)
        
    # Process data into DataFrame
    records = []
    best_model_name = ""
    best_r2 = -1.0
    best_params = {}
    
    for name, metrics in model_report.items():
        records.append({
            "Model Name": name,
            "R2 Score": round(metrics["r2_score"], 4),
            "MAE": round(metrics["mae"], 4),
            "MSE": round(metrics["mse"], 4),
            "RMSE": round(metrics["rmse"], 4)
        })
        if metrics["r2_score"] > best_r2:
            best_r2 = metrics["r2_score"]
            best_model_name = name
            best_params = metrics["best_params"]
            
    df_metrics = pd.DataFrame(records)
    
    # Highlight best model in card
    st.markdown(f"""
        <div class="highlight-card">
            <h3>🏆 Best Performing Model: {best_model_name}</h3>
            <p style="font-size: 1.15rem; margin: 5px 0;">This model achieved the highest predictive accuracy on our test partition with an <b>R² Score of {best_r2:.4f}</b>.</p>
            <p style="font-size: 0.95rem; opacity: 0.85; margin: 5px 0;"><b>Hyperparameters tuned:</b> {json.dumps(best_params)}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics Table
    st.subheader("📊 Performance Summary Table")
    
    # Highlight helper function
    def highlight_best(s):
        is_max = s == s.max() if s.name == 'R2 Score' else s == s.min()
        return ['background-color: rgba(39, 174, 96, 0.2); font-weight: bold' if v else '' for v in is_max]
        
    styled_df = df_metrics.style.apply(highlight_best, subset=['R2 Score', 'MAE', 'MSE', 'RMSE'])
    st.dataframe(styled_df, use_container_width=True)
    
    st.write("💡 *Green cells highlight the best performing model metrics (highest R², lowest MAE/MSE/RMSE).*")
    st.ln()
    
    # Visualizations
    st.subheader("📊 Interactive Metric Visualization")
    
    # Let user select which metric to visualize
    metric_choice = st.selectbox("Select Evaluation Metric to Compare", ["R2 Score", "MAE", "MSE", "RMSE"])
    
    # Sort DataFrame by chosen metric
    df_sorted = df_metrics.sort_values(by=metric_choice, ascending=(metric_choice != "R2 Score"))
    
    # Highlight color for best model bar
    colors = ['#2e59a8' if model != best_model_name else '#27ae60' for model in df_sorted["Model Name"]]
    
    fig = go.Figure(go.Bar(
        x=df_sorted["Model Name"],
        y=df_sorted[metric_choice],
        marker_color=colors,
        text=df_sorted[metric_choice],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Model Comparison by {metric_choice}",
        xaxis_title="Machine Learning Model",
        yaxis_title=metric_choice,
        height=450,
        font=dict(family="Inter, sans-serif")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
else:
    st.warning("Model training report not found. Please train models first.")
