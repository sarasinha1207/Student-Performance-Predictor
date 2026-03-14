import os
import numpy as np
import pandas as pd

def generate_student_dataset(output_path="data/student_data.csv", num_samples=1000, random_seed=42):
    np.random.seed(random_seed)
    
    # Generate Academic Features
    previous_score = np.random.normal(loc=68, scale=12, size=num_samples)
    previous_score = np.clip(previous_score, 40, 100)
    
    # Attendance is skewed high
    attendance = np.random.beta(a=5, b=1.5, size=num_samples) * 50 + 50
    attendance = np.clip(attendance, 50, 100)
    
    assignments_completed = np.random.binomial(n=10, p=0.8, size=num_samples)
    study_hours = np.random.gamma(shape=3, scale=1.5, size=num_samples)
    study_hours = np.clip(study_hours, 1, 10)
    
    class_participation = np.random.choice([1, 2, 3, 4, 5], size=num_samples, p=[0.1, 0.15, 0.3, 0.3, 0.15])
    
    # Failures (0-5 scale), heavily skewed towards 0 failures
    failures = np.random.choice([0, 1, 2, 3, 4, 5], size=num_samples, p=[0.65, 0.18, 0.08, 0.05, 0.03, 0.01])
    
    # Generate Lifestyle Features
    sleep_hours = np.random.normal(loc=7.0, scale=1.0, size=num_samples)
    sleep_hours = np.clip(sleep_hours, 4, 10)
    
    screen_time = np.random.gamma(shape=2, scale=2, size=num_samples)
    screen_time = np.clip(screen_time, 1, 10)
    
    physical_activity = np.random.binomial(n=5, p=0.5, size=num_samples)
    
    # Travel Time to School (commute ranges)
    travel_time_options = ["Under 15 min", "15-30 min", "30-60 min", "Over 60 min"]
    travel_time = np.random.choice(travel_time_options, size=num_samples, p=[0.4, 0.35, 0.18, 0.07])
    
    # Generate Demographic Features
    internet_access = np.random.choice(["Yes", "No"], size=num_samples, p=[0.85, 0.15])
    school_type = np.random.choice(["Public", "Private"], size=num_samples, p=[0.75, 0.25])
    
    # Map categorical values to numeric weights for target calculation
    travel_time_weights = {
        "Under 15 min": 0.0,
        "15-30 min": -0.5,
        "30-60 min": -1.5,
        "Over 60 min": -3.0
    }
    
    internet_weights = {"Yes": 2.0, "No": 0.0}
    school_weights = {"Public": 0.0, "Private": 2.0}
    
    # Calculate base target score
    final_score = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Academic contributions (weight ~ 80%)
        ac_contrib = (
            (previous_score[i] * 0.45) + 
            (attendance[i] * 0.25) + 
            (study_hours[i] * 1.5) + 
            (assignments_completed[i] * 1.0) + 
            (class_participation[i] * 1.2)
        )
        
        # Lifestyle contributions
        lf_contrib = (sleep_hours[i] * 0.4) - (screen_time[i] * 0.3) + (physical_activity[i] * 0.2)
        
        # Demographic, Failure, and Commute contributions
        failures_contrib = - (failures[i] * 4.5)
        travel_contrib = travel_time_weights[travel_time[i]]
        
        dm_contrib = (
            internet_weights[internet_access[i]] +
            school_weights[school_type[i]] +
            failures_contrib +
            travel_contrib
        )
        
        # Combine with some noise
        noise = np.random.normal(0, 2.5)
        
        # Scale adjustment to map values back to normal test ranges
        raw_final = ac_contrib + lf_contrib + dm_contrib + noise
        final_score[i] = raw_final
        
    # Standardize final_score to be between 45 and 100 based on standard academic performance
    final_score = np.clip(final_score, 45, 100)
    final_score = np.round(final_score).astype(int)
    
    # Construct DataFrame
    df = pd.DataFrame({
        "previous_score": np.round(previous_score, 1),
        "attendance": np.round(attendance, 1),
        "assignments_completed": assignments_completed,
        "study_hours": np.round(study_hours, 1),
        "class_participation": class_participation,
        "failures": failures,
        "sleep_hours": np.round(sleep_hours, 1),
        "screen_time": np.round(screen_time, 1),
        "physical_activity": physical_activity,
        "travel_time": travel_time,
        "internet_access": internet_access,
        "school_type": school_type,
        "final_score": final_score
    })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset saved to {output_path} with {num_samples} samples.")
    return df

if __name__ == "__main__":
    generate_student_dataset()
