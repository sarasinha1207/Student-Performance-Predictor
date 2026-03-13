# Student Performance Predictor

A Machine Learning web application designed to predict student academic outcomes and detect risk levels early using advanced predictive modeling. The application is built with Streamlit and compares 7 distinct regressor models to output the most accurate predictions.

**Live Deployment Link**: https://student-performance-predictor-07.streamlit.app/

<p align="center"><img width="1000" height="800" alt="image" src="https://github.com/user-attachments/assets/e5af054a-65b8-430a-af13-a5d9357a94c7" /></p>

---

## Key Features

* **Advanced Predictive Modeling**: Inputs academic parameters (attendance, previous scores, assignments, study hours) and lifestyle factors (sleep, screen time, physical activity) to predict final exam scores.
* **Explainable AI (XAI) & Dynamic Advisory**: Generates real-time, personalized recommendations for students based on their specific lifestyle and academic risk distributions (e.g., attendance drops combined with low assignment rates).
* **What-If Analysis Dashboard**: Allows educators and students to slide model parameters dynamically and watch expected scores shift on an interactive speedometer/gauge chart in real-time.
* **Model Evaluation & Leaderboard**: Ranks all 7 regressor models based on cross-validated R² score, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
* **Visual Analytics**: Interactive Plotly scatter plots and heatmaps highlighting attendance, study hours, correlation matrices, and relative feature importances.

## Machine Learning Pipeline

<p align="center"><img width="450" height="720" alt="image" src="https://github.com/user-attachments/assets/cfa66763-ec4f-4ae1-b763-1a3c8a9be331" /></p>

### 1. Data Ingestion & Generation
* Generates a synthetic dataset (`student_data.csv`) of 1,000 students modeled on realistic student distributions (skewed attendance, failures heavily weighted at 0, gamma-distributed study hours, sleep/screen time inverse correlations).
* Splits the dataset into an 80% training set and 20% testing set.

### 2. Data Transformation
* **Numerical Features Pipeline**: Mediated imputation of missing values, followed by standard scaling (`StandardScaler`) to normalize distribution ranges.
  * *Features*: Previous Score, Attendance, Assignments Completed, Study Hours, Class Participation, Failures, Sleep Hours, Screen Time, Physical Activity.
* **Categorical Features Pipeline**: Frequency-imputed missing values, followed by `OneHotEncoder` and standard scaling.
  * *Features*: Travel Time, Internet Access, School Type.
* Saves the resulting preprocessor pipeline binary to `models/preprocessor.joblib`.

### 3. Model Training & Parameter Tuning
* Compares 7 machine learning regressors: Linear Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, and CatBoost.
* Uses `GridSearchCV` with 3-fold cross-validation to search the parameter space for the optimal configuration of each model.
* Saves the best-performing model binary (`Linear Regression` in the default run) to `models/model.joblib`.
* Extracts a subset of training data to `models/background_data.joblib` to facilitate SHAP model interpretation.


## ML Model Comparison & Performance Metrics

Below are the evaluation metrics of the trained models sorted by their $R^2$ performance on the test set:

| Rank | Model | $R^2$ Score | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) | Best Parameters |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 1. | **Linear Regression** | **0.9181** | **2.0760** | **6.7719** | **2.6023** | *N/A (Standard)* |
| 2. | **CatBoosting Regressor** | 0.8831 | 2.4480 | 9.6644 | 3.1088 | `depth: 6`, `iterations: 100`, `learning_rate: 0.1` |
| 3. | **Gradient Boosting** | 0.8747 | 2.5022 | 10.3605 | 3.2188 | `learning_rate: 0.1`, `n_estimators: 128`, `subsample: 0.8` |
| 4 | **XGBRegressor** | 0.8647 | 2.6108 | 11.1884 | 3.3449 | `learning_rate: 0.1`, `max_depth: 3`, `n_estimators: 128` |
| 5 | **Random Forest** | 0.8221 | 3.1029 | 14.7076 | 3.8351 | `max_depth: 10`, `n_estimators: 256` |
| 6 | **AdaBoost Regressor** | 0.7661 | 3.5155 | 19.3324 | 4.3969 | `learning_rate: 0.5`, `n_estimators: 256` |
| 7 | **Decision Tree** | 0.6448 | 4.2065 | 29.3605 | 5.4185 | `criterion: poisson`, `max_depth: 8` |

<p align="center"><img width="700" height="700" alt="image" src="https://github.com/user-attachments/assets/37df071d-60ff-4eee-8fda-59c05eec11f6" /></p>


## Installation and Run Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/sarasinha1207/Student-Performance-Predictor.git
cd Student-Performance-Predictor
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute the ML Pipeline
This command triggers synthetic data generation, preprocessing, model hyperparameter tuning, model ranking, and serializes the pipeline objects:
```bash
python -m src.components.data_ingestion
```

### 5. Launch the Streamlit App
Run the local development server:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.


## Project Structure

```text
├── .streamlit/
│   └── config.toml         # Custom Streamlit theme locks (Indigo/Cyan accents)
├── data/
│   ├── student_data.csv    # Generated raw dataset (1000 samples)
│   ├── train.csv           # Ingested training split (80%)
│   └── test.csv            # Ingested testing split (20%)
├── models/
│   ├── preprocessor.joblib # Serialized data prep pipeline
│   ├── model.joblib        # Serialized best trained model
│   ├── background_data.joblib # Subset for XAI reference
│   └── model_report.json   # Model evaluation scores
├── notebooks/
│   └── EDA.ipynb           # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── exception.py        # Custom logging exception handler
│   ├── utils.py            # Helper scripts (evaluate, save, load objects)
│   ├── components/
│   │   ├── generate_data.py # Data simulation script
│   │   ├── data_ingestion.py # Splits data and triggers components
│   │   ├── data_transformation.py # Preprocessing pipeline
│   │   └── model_trainer.py # Fits model & executes Grid Search
│   └── pipeline/
│       ├── predict_pipeline.py # Model serving utility class
│       └── train_pipeline.py
├── app.py                  # Main Streamlit web application
├── setup.py                # Package building instructions
└── requirements.txt        # Library dependencies
```
