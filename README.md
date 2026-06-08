# Student Performance Predictor

This project is a Flask-based web application that predicts student performance based on various input parameters. It utilizes machine learning models to provide insights into potential student outcomes.

## Features

- Predict student performance based on:
    - Gender
    - Ethnicity
    - Parental Education
    - Lunch Type
    - Test Preparation
    - Reading Score
    - Writing Score
- User-friendly web interface for inputting parameters and viewing predictions.
- Machine learning model for accurate predictions.

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yashg8180/mlproject.git
    cd mlproject
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask application:**

    ```bash
    python app.py
    ```

    The application will typically run on `http://127.0.0.1:5000/`.

## Project Structure

-   `app.py`: Main Flask application file.
-   `templates/`: Contains HTML templates (`index.html`, `home.html`).
-   `artifacts/`: Stores trained machine learning models and preprocessors.
-   `src/`: Contains source code for data ingestion, transformation, model training, and prediction pipelines.
-   `notebook/`: Jupyter notebooks for EDA and model training.
-   `requirements.txt`: Lists project dependencies.

## Usage

Navigate to the application in your web browser, input the required student parameters, and get a prediction of their performance.