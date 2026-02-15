# Generate README.md content for ML Assignment 2: Campus Placement Prediction (Classification Only)

readme_md = """# ML Assignment 2: Campus Placement Prediction (Classification Only)

## a. Problem Statement

Campus placements are a critical milestone for students and institutions. Predicting whether a student will be placed helps universities and training teams optimize interventions, personalize skill development, and improve placement outcomes.

Objectives:
- Build and compare six classification models to predict whether a student is placed.
- Evaluate each model using Accuracy, AUC, Precision, Recall, F1, and MCC.
- Identify the most effective and balanced model for deployment.
- Provide an interactive Streamlit app for dataset exploration, training analysis (with confusion matrix near heatmap), and batch predictions via CSV upload.

Business impact:
- Early prediction enables targeted skill training and interview readiness.
- Better resource allocation towards students who need support.
- Improves institutional KPIs and student outcomes.

## b. Dataset Description

Dataset: Campus Placement (synthetic)

Overview (from the Data Overview tab):
- Total Students: 100,000
- Features (including target): 26
- Placement Rate: 84.4%
- Missing Values: 64,965 (handled during preprocessing)

Feature Categories:
- Demographics: gender, age, city_tier
- Academics:
  - ssc_percentage, ssc_board
  - hsc_percentage, hsc_board, hsc_stream
  - degree_percentage, degree_field
  - mba_percentage, specialization
- Experience & Achievements: internships_count, projects_count, certifications_count, work_experience_months
- Skills: technical_skills_score, soft_skills_score, aptitude_score, communication_score
- Activities: leadership_roles, extracurricular_activities, backlogs
- Target: placed (Binary: 1 = Placed, 0 = Not placed)
- Note: salary_lpa exists but is excluded from classification to avoid leakage.

Preprocessing Steps:
- Dropped non-predictive identifiers: student_id.
- Dropped leakage feature: salary_lpa from classification features.
- One-hot encoded categorical variables (drop_first=True).
- Normalized binary-like fields to 0/1 (e.g., leadership_roles, extracurricular_activities).
- Filled missing numeric values with medians.
- Applied Standard Scaling for models requiring normalization (Logistic Regression, kNN).

## c. Models Used

Six classification models were implemented and evaluated using a 70-30 train-test split with stratification to maintain class distribution.

### Comparison Table (rounded to 4 decimals)

| ML Model Name              | Accuracy | AUC    | Precision | Recall  | F1     | MCC    |
|---------------------------|----------|--------|-----------|---------|--------|--------|
| Logistic Regression       | 0.8593   | 0.8417 | 0.8761    | 0.9707  | 0.9209 | 0.3334 |
| Decision Tree             | 0.8428   | 0.7654 | 0.8717    | 0.9543  | 0.9111 | 0.2636 |
| kNN                       | 0.8394   | 0.6903 | 0.8614    | 0.9651  | 0.9103 | 0.1971 |
| Naive Bayes               | 0.8337   | 0.8179 | 0.9014    | 0.9016  | 0.9015 | 0.3669 |
| Random Forest (Ensemble)  | 0.8558   | 0.8263 | 0.8645    | 0.9834  | 0.9201 | 0.2737 |
| XGBoost (Ensemble)        | 0.8487   | 0.8080 | 0.8785    | 0.9525  | 0.9140 | 0.3100 |

## d. Observations on Model Performance

| ML Model Name              | Observation about model performance                                                                                                                         |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression       | Strong overall performance with high AUC (0.8417) and F1 (0.9209). Very high recall (0.9707) with solid precision (0.8761). Interpretable and deployment-friendly. |
| Decision Tree             | Good recall (0.9543) and decent precision (0.8717), but lower AUC (0.7654) suggests weaker discriminative power. Can benefit from tuning (depth, min samples).    |
| kNN                       | High recall (0.9651) but lower AUC (0.6903) and MCC (0.1971). Sensitive to scaling and high-dimensional space; prediction latency grows with dataset size.        |
| Naive Bayes               | Most balanced precision/recall (~0.901 each) and highest MCC (0.3669), indicating better balance under class imbalance. Accuracy slightly lower (0.8337).        |
| Random Forest (Ensemble)  | Excellent recall (0.9834) and high F1 (0.9201). Good AUC (0.8263). Slightly conservative precision (0.8645); provides useful feature importance for insights.      |
| XGBoost (Ensemble)        | Strong balance with high precision (0.8785), recall (0.9525), and F1 (0.9140). AUC (0.8080) shows solid discriminative power. Efficient and production-friendly.   |

Key Insights:
- High recall across models implies strong sensitivity to the placed class—useful when minimizing false negatives (placed predicted as not placed).
- Naive Bayes shows the highest MCC, indicating better balance under class imbalance.
- Random Forest and Logistic Regression deliver top F1 scores for practical effectiveness.
- Class imbalance (84.4% placed) should be considered; threshold tuning, class weights, or calibration may help if the minority class (not placed) is the focus.

## Streamlit Application Features

- Dataset Upload Option (CSV): Upload test data for batch predictions.
- Model Selection Dropdown: Choose among 6 trained models.
- Display of Evaluation Metrics: Accuracy, AUC, Precision, Recall, F1, MCC.
- Confusion Matrix and Classification Report: Displayed when true labels exist in uploaded data; confusion matrix also shown in Training Analysis near heatmap.
- Dataset Description: Data overview, quick visuals, statistical summaries (numerical/categorical), and dataset download buttons.

## Repository Structure

- main.py (Streamlit app)
- train.py (Classification-only training script)
- models/
  - scaler_classification.pkl
  - feature_names_classification.pkl
  - logistic_regression.pkl
  - decision_tree.pkl
  - knn.pkl
  - naive_bayes.pkl
  - random_forest.pkl
  - xgboost.pkl
  - model_comparison_classification.csv
- data/
  - dataset.csv
  - test.csv (optional for demo)
- requirements.txt
- README.md

## Installation and Usage

Prerequisites:
- Python 3.8+ and pip

Install dependencies:
- pip install -r requirements.txt

Train models (classification-only):
- python train.py

Run the Streamlit app locally:
- streamlit run main.py

Access the app:
- Open your browser and go to http://localhost:8501

## Acknowledgements

Thanks to contributors and synthetic dataset providers. This project demonstrates end-to-end ML workflow—modeling, evaluation, UI/UX, and deployment—aligned with ML Assignment 2 requirements.
"""

# Write README.md to disk
from pathlib import Path

output_path = Path("README.md")
output_path.write_text(readme_md, encoding="utf-8")

print(f"README.md has been written to: {output_path.resolve()}")