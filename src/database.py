import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join("data", "predictions.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            predicted_score REAL NOT NULL,
            grade TEXT NOT NULL,
            risk_category TEXT NOT NULL,
            model_used TEXT NOT NULL,
            study_hours REAL NOT NULL,
            attendance REAL NOT NULL,
            previous_score REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(student_name, predicted_score, grade, risk_category, model_used, study_hours, attendance, previous_score):
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO prediction_history (
            student_name, timestamp, predicted_score, grade, risk_category, model_used, study_hours, attendance, previous_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (student_name, timestamp, predicted_score, grade, risk_category, model_used, study_hours, attendance, previous_score))
    conn.commit()
    conn.close()

def get_prediction_history():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prediction_history ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def clear_prediction_history():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history")
    conn.commit()
    conn.close()
