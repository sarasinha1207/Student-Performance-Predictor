# Student Performance Predictor

This project is a production-ready Streamlit web application that predicts a student's final academic performance based on demographic, lifestyle, and academic features. It compares 7 regression models using cross-validated Grid Search to select the best predictor.

## Features

- Predict student final exam scores based on:
    - Academic Factors: Previous Score, Attendance Rate, Assignments Completed, Study Hours, Class Participation.
    - Lifestyle Factors: Sleep Hours, Screen Time, Physical Activity.
    - Demographic Factors: Parent Education Level, Family Income Status, Internet Access, School Type.
- What-If Analysis Module: Slide parameters in real-time and view score outcomes on an interactive gauge chart.
- Performance Comparison: View metrics (R2 Score, MAE, MSE, RMSE) of all 7 machine learning models.
- Analytics Dashboard: Scatter plots showing attendance and study patterns relative to final scores.
- Memory Caching: Model and preprocessor components are cached for instant predictions.

## Setup and Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/sarasinha1207/Student-Performance-Predictor.git
    cd Student-Performance-Predictor
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    venv\Scripts\activate  # On macOS/Linux: source venv/bin/activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Train the ML models:**

    ```bash
    python -m src.components.data_ingestion
    ```

5. **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

    The application will run on `http://localhost:8501`.

## Project Structure

- `app.py`: Main Streamlit web application.
- `data/`: Stores raw, training, and testing datasets.
- `models/`: Stores serialized joblib models and model transformation preprocessors.
- `notebooks/`: Contains the Jupyter notebook for exploratory data analysis (EDA).
- `src/`: Contains pipeline source code (data ingestion, data transformation, model training, and prediction pipeline).
- `requirements.txt`: Lists project dependencies.
- `setup.py`: Project setup and package installation script.