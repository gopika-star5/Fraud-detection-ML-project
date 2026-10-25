# Industry-Oriented Machine Learning System for Fraud Detection

## Objective
Detect suspicious transactions using machine learning.

## Dataset
The ZIP contains a self-contained synthetic demonstration dataset with 10,000 transactions so the project is runnable without downloading another file.

## Methodology
Data cleaning → feature engineering → stratified split → imputation → scaling → SMOTE on training data → Logistic Regression / Decision Tree / Random Forest → Precision / Recall / F1-score / ROC-AUC.

## Results

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Random Forest | 0.974 | 0.958 | 0.966 | 1.000 |
| Logistic Regression | 0.913 | 0.975 | 0.943 | 0.998 |
| Decision Tree | 0.898 | 0.958 | 0.927 | 0.957 |

**Best model in the included demo:** Random Forest.

## Important
These results are from the included synthetic dataset. For a final academic submission, replace the demo dataset with an approved real/public dataset if your internship requires it and rerun the notebook.

## Future Scope
Real-time monitoring, explainable AI, threshold optimization, API deployment and model drift monitoring.
