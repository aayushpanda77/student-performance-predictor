"""
generate_data.py
----------------
Generates a realistic synthetic dataset for student performance prediction.
NOTE: This dataset is synthetic and created strictly for educational and demonstration purposes.
It does not contain real personal records of any students.
"""

import numpy as np
import pandas as pd
import os

def generate_student_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic student performance data with realistic statistical distributions.
    
    Parameters:
        n_samples (int): Number of student records to generate (default: 1000).
        random_state (int): Random seed for reproducibility (default: 42).
        
    Returns:
        pd.DataFrame: DataFrame containing features and target variable.
    """
    np.random.seed(random_state)
    
    # 1. Study hours per day: normal distribution clipped between 1.0 and 10.0 hours
    study_hours = np.random.normal(loc=5.2, scale=1.8, size=n_samples)
    study_hours = np.clip(study_hours, 1.0, 10.0)
    
    # 2. Attendance percentage: beta distribution scaled between 55% and 100%
    # Skewed towards higher attendance, which is typical in university environments
    attendance = 55.0 + 45.0 * np.random.beta(a=5, b=2, size=n_samples)
    attendance = np.clip(attendance, 55.0, 100.0)
    
    # 3. Assignment / Internal test score: influenced by study habits + random variation
    # Clipped between 40.0 and 100.0
    assignment_base = 45.0 + (study_hours / 10.0) * 35.0 + (attendance / 100.0) * 15.0
    assignment_noise = np.random.normal(loc=0.0, scale=6.0, size=n_samples)
    assignment_score = np.clip(assignment_base + assignment_noise, 40.0, 100.0)
    
    # 4. Sleep hours per day: normal distribution clipped between 4.5 and 9.5 hours
    sleep_hours = np.random.normal(loc=6.8, scale=1.1, size=n_samples)
    sleep_hours = np.clip(sleep_hours, 4.5, 9.5)
    
    # 5. Final Score (Target): Linear combination of academic factors + noise
    # Weights reflect reasonable domain expectations:
    # - Assignment score: ~37% weight
    # - Study hours: ~3.15 points per daily hour
    # - Attendance: ~0.26 points per percentage point
    # - Sleep hours: ~1.20 points per hour of restful sleep
    final_score_raw = (
        6.0
        + (3.15 * study_hours)
        + (0.26 * attendance)
        + (0.37 * assignment_score)
        + (1.20 * sleep_hours)
        + np.random.normal(loc=0.0, scale=2.8, size=n_samples)
    )
    final_score = np.clip(final_score_raw, 0.0, 100.0)
    
    # Construct DataFrame with rounded values for realistic presentation
    df = pd.DataFrame({
        'study_hours_per_day': np.round(study_hours, 1),
        'attendance_percent': np.round(attendance, 1),
        'assignment_score': np.round(assignment_score, 1),
        'sleep_hours': np.round(sleep_hours, 1),
        'final_score': np.round(final_score, 1)
    })
    
    return df

def main():
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "student_performance.csv")
    
    print("Generating synthetic student performance dataset...")
    df = generate_student_dataset(n_samples=1000, random_state=42)
    df.to_csv(output_path, index=False)
    
    print(f"[SUCCESS] Dataset successfully generated at: {output_path}")
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst 5 records:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())

if __name__ == "__main__":
    main()
