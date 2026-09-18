"""
train_model.py
--------------
Trains a Supervised Machine Learning model (Linear Regression) to predict student academic performance.
Performs data loading, inspection, preprocessing, model training, evaluation, and visualization.

Author: Ayush Panda
Institution: SRM Institute of Science and Technology, Chennai
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def create_visualizations(df: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, output_dir: str):
    """
    Generate and save exploratory data analysis (EDA) and evaluation charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    
    # 1. Study Hours vs Final Score
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df,
        x='study_hours_per_day',
        y='final_score',
        scatter_kws={'alpha': 0.6, 'color': '#1f77b4'},
        line_kws={'color': '#d62728', 'linewidth': 2}
    )
    plt.title("Study Hours per Day vs Final Score", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Study Hours per Day", fontsize=11)
    plt.ylabel("Final Score (out of 100)", fontsize=11)
    plt.tight_layout()
    plot_path1 = os.path.join(output_dir, "study_hours_vs_score.png")
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"  [SAVED] {plot_path1}")

    # 2. Attendance vs Final Score
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df,
        x='attendance_percent',
        y='final_score',
        scatter_kws={'alpha': 0.6, 'color': '#2ca02c'},
        line_kws={'color': '#d62728', 'linewidth': 2}
    )
    plt.title("Attendance Percentage vs Final Score", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Attendance Percentage (%)", fontsize=11)
    plt.ylabel("Final Score (out of 100)", fontsize=11)
    plt.tight_layout()
    plot_path2 = os.path.join(output_dir, "attendance_vs_score.png")
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"  [SAVED] {plot_path2}")

    # 3. Assignment Score vs Final Score
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df,
        x='assignment_score',
        y='final_score',
        scatter_kws={'alpha': 0.6, 'color': '#ff7f0e'},
        line_kws={'color': '#d62728', 'linewidth': 2}
    )
    plt.title("Assignment Score vs Final Score", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Assignment Score", fontsize=11)
    plt.ylabel("Final Score (out of 100)", fontsize=11)
    plt.tight_layout()
    plot_path3 = os.path.join(output_dir, "assignment_vs_score.png")
    plt.savefig(plot_path3, dpi=300)
    plt.close()
    print(f"  [SAVED] {plot_path3}")

    # 4. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr_matrix = df.corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Heatmap of Student Factors", fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plot_path4 = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(plot_path4, dpi=300)
    plt.close()
    print(f"  [SAVED] {plot_path4}")

    # 5. Distribution of Final Scores
    plt.figure(figsize=(8, 5))
    sns.histplot(
        df['final_score'],
        kde=True,
        color='#9467bd',
        bins=25,
        edgecolor='black'
    )
    plt.axvline(df['final_score'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['final_score'].mean():.2f}")
    plt.axvline(df['final_score'].median(), color='green', linestyle=':', linewidth=2, label=f"Median: {df['final_score'].median():.2f}")
    plt.title("Distribution of Final Academic Scores", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Final Score", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plot_path5 = os.path.join(output_dir, "score_distribution.png")
    plt.savefig(plot_path5, dpi=300)
    plt.close()
    print(f"  [SAVED] {plot_path5}")

    # 6. Actual vs Predicted Scores (Evaluation Plot)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, color='#17becf', edgecolors='k', linewidth=0.5, label='Predictions')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y = x)')
    plt.title("Actual vs Predicted Final Scores (Test Set)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Actual Score", fontsize=11)
    plt.ylabel("Predicted Score", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plot_path6 = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(plot_path6, dpi=300)
    plt.close()
    print(f"  [SAVED] {plot_path6}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "student_performance.csv")
    model_dir = os.path.join(base_dir, "model")
    images_dir = os.path.join(base_dir, "images")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print("=" * 65)
    print("      STUDENT PERFORMANCE PREDICTOR - MODEL TRAINING PIPELINE")
    print("=" * 65)

    # STEP 1: Load Dataset
    print("\n[STEP 1] Loading dataset...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_data.py first.")
    
    df = pd.read_csv(data_path)
    print(f"  Dataset loaded successfully! Total records: {len(df)}")
    print(f"  Features present: {list(df.columns)}")

    # STEP 2: Data Cleaning & Inspection
    print("\n[STEP 2] Inspecting and verifying data cleanliness...")
    missing_count = df.isnull().sum().sum()
    print(f"  Total missing/null values: {missing_count}")
    duplicate_count = df.duplicated().sum()
    print(f"  Total duplicate rows: {duplicate_count}")
    
    print("\n  Summary Statistics:")
    print(df.describe().to_string())

    # STEP 3: Separate Features and Target
    print("\n[STEP 3] Separating input features (X) and target variable (y)...")
    feature_columns = ['study_hours_per_day', 'attendance_percent', 'assignment_score', 'sleep_hours']
    target_column = 'final_score'

    X = df[feature_columns]
    y = df[target_column]

    print(f"  Input Features (X): {feature_columns}")
    print(f"  Target Variable (y): '{target_column}'")

    # STEP 4: Train / Test Split
    print("\n[STEP 4] Splitting dataset into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"  Training set size: {X_train.shape[0]} samples")
    print(f"  Testing set size:  {X_test.shape[0]} samples")

    # STEP 5: Train Linear Regression Model
    print("\n[STEP 5] Training Linear Regression model on training data...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("  Model training complete!")

    # STEP 6: Make Predictions on Test Set
    print("\n[STEP 6] Making predictions on the unseen test set...")
    y_pred = model.predict(X_test)

    # STEP 7: Evaluate Model Performance
    print("\n[STEP 7] Evaluating model performance...")
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("-" * 50)
    print("               MODEL EVALUATION RESULTS")
    print("-" * 50)
    print(f"  Mean Absolute Error (MAE) : {mae:.4f}")
    print(f"  Mean Squared Error (MSE)  : {mse:.4f}")
    print(f"  Root Mean Squared Error   : {rmse:.4f}")
    print(f"  R-squared (R2 Score)      : {r2:.4f} (or {r2 * 100:.2f}%)")
    print("-" * 50)

    print("\nModel Coefficients (Feature Weights):")
    for feat, coef in zip(feature_columns, model.coef_):
        print(f"  - {feat:22s} : +{coef:.4f}")
    print(f"  - Intercept (beta_0)      : {model.intercept_:.4f}")

    # STEP 8: Save Trained Model and Metrics
    print("\n[STEP 8] Saving trained model and metadata...")
    model_file = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_file)
    print(f"  [SAVED] Model saved to: {model_file}")

    metrics_data = {
        "algorithm": "Linear Regression",
        "features": feature_columns,
        "target": target_column,
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "r2_score": round(float(r2), 4),
        "coefficients": {feat: round(float(c), 4) for feat, c in zip(feature_columns, model.coef_)},
        "intercept": round(float(model.intercept_), 4)
    }

    metrics_file = os.path.join(model_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"  [SAVED] Metrics metadata saved to: {metrics_file}")

    # STEP 9: Create and Save Visualizations
    print("\n[STEP 9] Generating and saving data visualizations...")
    create_visualizations(df, y_test, y_pred, images_dir)

    print("\n" + "=" * 65)
    print("      TRAINING & EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
