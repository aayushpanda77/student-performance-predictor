# 🎓 Student Performance Predictor

An end-to-end Supervised Machine Learning web application that predicts a student's expected final academic score based on study habits, attendance, test performance, and sleep duration.

Built with **Python**, **Scikit-learn**, **Pandas**, **Matplotlib**, **Seaborn**, and **Streamlit**.

---

## 📌 Project Overview
Academic performance is influenced by daily study routines, consistent attendance, internal assessments, and healthy lifestyle habits. The **Student Performance Predictor** models these relationships using **Linear Regression** to estimate a student's final score (out of 100). The project includes a full machine learning pipeline—from exploratory data analysis and model training to an interactive, user-friendly Streamlit web interface.

---

## 🎯 Problem Statement
Students and academic advisors often lack a quick, quantifiable way to estimate how daily routines impact final course outcomes. Without timely feedback early or midway through a semester, students may not realize that poor attendance or irregular study schedules are pulling down their grades until final exam results are published.

---

## 🚀 Objective
1. Build a clean, reproducible machine learning pipeline that predicts continuous academic scores from key behavioral and academic features.
2. Provide transparent, explainable machine learning predictions using Linear Regression.
3. Deploy an intuitive web application where students can simulate different study scenarios and receive instant score predictions with actionable interpretations.

---

## ✨ Features
- **Exploratory Data Analysis (EDA):** Visualizations illustrating correlations, distributions, and feature trends with final scores.
- **Supervised ML Model:** Ordinary Least Squares (OLS) Linear Regression with train-test split validation (80/20).
- **Comprehensive Evaluation:** Real performance tracking using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared ($R^2$).
- **Interactive Web UI:** Modern, clean Streamlit application featuring number input controls, real-time score calculation, and performance categorization (Excellent, Good, Needs Improvement).
- **In-App Chart Explorer:** View correlation heatmaps and regression plots directly inside the web interface.
- **Model Persistence:** Serialized model export and loading using `joblib`.

---

## 🛠️ Technologies Used
| Category | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.x | Core programming language |
| **Data Processing** | Pandas, NumPy | Data manipulation, cleaning, and array computations |
| **Machine Learning** | Scikit-learn | Data splitting, model training, and metric evaluations |
| **Data Visualization** | Matplotlib, Seaborn | Distribution plots, regression plots, and correlation heatmap |
| **Web Framework** | Streamlit | Interactive web application and dashboard |
| **Model Persistence** | Joblib | Model serialization (`model.pkl`) and deserialization |

---

## 📊 Dataset Description
> **IMPORTANT NOTICE:**  
> The dataset (`data/student_performance.csv`) is **synthetic and educational**. It was programmatically generated using realistic statistical distributions for educational demonstration and does **NOT** contain real, private student records.

The dataset comprises **1,000 student records** across 5 continuous variables:

| Feature Name | Type | Unit / Range | Description |
| :--- | :--- | :--- | :--- |
| `study_hours_per_day` | Float | 1.0 – 10.0 hrs | Average hours spent in self-study daily |
| `attendance_percent` | Float | 55.0% – 100.0% | Percentage of lectures attended during the semester |
| `assignment_score` | Float | 40.0 – 100.0 | Cumulative score on quizzes, assignments, and tests |
| `sleep_hours` | Float | 4.5 – 9.5 hrs | Average daily sleep duration |
| **`final_score` (Target)** | Float | 0.0 – 100.0 | Final academic score obtained (continuous target) |

---

## 🧠 Machine Learning Algorithm: Linear Regression
The project employs **Linear Regression**, a supervised learning algorithm for continuous numerical targets. It fits a linear equation that minimizes the sum of squared residuals between observed and predicted scores.

### Mathematical Formulation
$$\hat{y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \beta_4 X_4$$

Where:
- $\hat{y}$ = Predicted Final Score
- $\beta_0$ = Intercept (baseline score)
- $\beta_1, \beta_2, \beta_3, \beta_4$ = Coefficients for study hours, attendance, assignments, and sleep
- $X_1, X_2, X_3, X_4$ = Student input values

### Learned Model Coefficients
From training on 800 samples:
- **Intercept ($\beta_0$):** `7.3621`
- **Study Hours ($\beta_1$):** `+2.9873` (each additional study hour adds ~2.99 marks)
- **Attendance % ($\beta_2$):** `+0.2407` (each 10% increase in attendance adds ~2.41 marks)
- **Assignment Score ($\beta_3$):** `+0.3771` (each 10-point gain in assignments adds ~3.77 marks)
- **Sleep Hours ($\beta_4$):** `+1.2546` (adequate sleep contributes ~1.25 marks per hour)

---

## 📈 Model Evaluation Metrics
The model was tested on 200 unseen records (20% test set). All metrics are calculated directly from model evaluation:

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Mean Absolute Error (MAE)** | **2.3739** | On average, predictions are within ~2.37 marks of actual scores. |
| **Mean Squared Error (MSE)** | **8.9887** | Average squared deviation across test samples. |
| **Root Mean Squared Error (RMSE)** | **2.9981** | Standard deviation of residuals in original score units. |
| **R² Score (Coefficient of Determination)** | **0.8743 (87.43%)** | 87.43% of the variance in final scores is explained by the input features. |

---

## 📁 Project Structure
```text
student-performance-predictor/
│
├── app.py                      # Streamlit interactive web application
├── train_model.py              # ML training, evaluation, and chart generation script
├── generate_data.py            # Reproducible synthetic dataset generator
├── requirements.txt            # Project dependencies
├── README.md                   # Comprehensive project documentation
├── INTERVIEW_PREPARATION.md    # 25 Q&As for technical interviews
├── .gitignore                  # Git ignore rules for clean repository
│
├── data/
│   └── student_performance.csv # Generated dataset (1,000 records)
│
├── model/
│   ├── model.pkl               # Serialized trained Scikit-learn model
│   └── metrics.json            # Evaluated metrics metadata
│
├── notebooks/
│   └── analysis.ipynb          # Jupyter notebook with step-by-step EDA and modeling
│
└── images/                     # Generated charts and visualizations
    ├── study_hours_vs_score.png
    ├── attendance_vs_score.png
    ├── assignment_vs_score.png
    ├── correlation_heatmap.png
    ├── score_distribution.png
    └── actual_vs_predicted.png
```

---

## 💻 Installation Instructions

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git (optional, for cloning)

### Step 1: Clone or Navigate to the Project Folder
```bash
git clone https://github.com/your-username/student-performance-predictor.git
cd student-performance-predictor
```

### Step 2: Create and Activate a Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### 1. (Optional) Re-generate Dataset
To regenerate the dataset from scratch:
```bash
python generate_data.py
```

### 2. Train the Model & Generate Visualizations
To train the Linear Regression model, evaluate its performance, and save the charts into `images/`:
```bash
python train_model.py
```

### 3. Launch the Streamlit Web Application
To run the interactive web interface:
```bash
streamlit run app.py
```
After running this command, open your web browser and navigate to:
```text
http://localhost:8501
```

---

## 💡 Example Usage
1. Open the web interface at `http://localhost:8501`.
2. Enter student attributes:
   - **Study Hours per Day:** `6.5`
   - **Attendance Percentage:** `90%`
   - **Assignment Score:** `82`
   - **Sleep Hours:** `7.5`
3. Click the **"Predict Score"** button.
4. **Output:**
   - **Predicted Score:** `88.58 / 100`
   - **Category:** `Excellent` 🌟
   - **Interpretation:** "The student shows strong academic habits and understanding. Consistent practice will help maintain top performance!"

---

## 🖼️ Exploratory Visualizations
The model training script automatically produces the following charts saved in the `images/` directory:

| Visualization | Description | File |
| :--- | :--- | :--- |
| **Correlation Heatmap** | Shows positive linear correlations across all academic features. | `images/correlation_heatmap.png` |
| **Study Hours vs Score** | Highlights strong upward trend with daily study dedication. | `images/study_hours_vs_score.png` |
| **Attendance vs Score** | Demonstrates the positive impact of lecture attendance. | `images/attendance_vs_score.png` |
| **Assignment vs Score** | Correlates internal test consistency with final exam results. | `images/assignment_vs_score.png` |
| **Score Distribution** | Illustrates Gaussian-like distribution around the mean (~81.3). | `images/score_distribution.png` |
| **Actual vs Predicted** | Compares model predictions against ground truth along the $y=x$ line. | `images/actual_vs_predicted.png` |

---

## 🔮 Future Improvements
1. **Polynomial & Interaction Features:** Model potential non-linear relationships, such as fatigue from excessive study hours without adequate sleep.
2. **Model Comparison:** Implement and compare other regression algorithms (Ridge, Lasso, Decision Trees, and Random Forests).
3. **Categorical Factors:** Incorporate factors such as study methods (group vs. solo study), course difficulty, and extracurricular participation.
4. **Cloud Deployment:** Deploy the application to Streamlit Cloud or containerize using Docker for cloud hosting on Render or AWS.

---

## 📚 What I Learned
- **End-to-End Machine Learning Pipeline:** Managing the workflow from raw data ingestion to user interface deployment.
- **Exploratory Data Analysis (EDA):** Identifying feature distributions, detecting outliers, and interpreting correlation coefficients using Matplotlib and Seaborn.
- **Model Training & Evaluation:** Understanding the practical trade-offs between model complexity, explainability, and evaluation metrics (MAE, MSE, RMSE, and $R^2$).
- **Web Deployment with Streamlit:** Creating an intuitive, responsive frontend for machine learning models using Python.
- **Code Cleanliness & Documentation:** Structuring a modular, readable codebase suitable for production and collaboration.

---

## 📄 Resume Entry
You can include the following bullet points on your resume:

**Student Performance Predictor | Python, Pandas, Scikit-learn, Streamlit**
- Developed a supervised machine learning model using Linear Regression to predict student final scores based on study hours, attendance, assignment performance, and sleep duration.
- Performed end-to-end data preprocessing, exploratory data analysis, feature correlation analysis, and model evaluation, achieving an $R^2$ score of 87.4% and an MAE of 2.37 marks.
- Built and deployed an interactive Streamlit web application providing instant score predictions, performance categories (Excellent, Good, Needs Improvement), and embedded data visualizations.

---

## 👤 Author
**Ayush Panda**  
B.Tech Artificial Intelligence  
SRM Institute of Science and Technology, Chennai  

---

## ⚖️ Disclaimer
*This project and its predictions are intended solely for educational and demonstration purposes. It should not be used as an official academic grading or assessment system.*
