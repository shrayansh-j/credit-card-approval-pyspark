# Credit Card Approval using Machine Learning in PySpark

##  Project Overview
Banks receive thousands of credit card applications every day. Approving high-risk applicants leads to financial loss, while rejecting good customers impacts revenue and customer experience.

This project builds an end-to-end machine learning pipeline using **Azure Databricks and PySpark** to predict whether a credit card applicant is a **good** or **bad** client based on demographic and credit history data.

---

## Objective
Build a machine learning model that classifies applicants as:
- 0 → Safe applicant  
- 1 → Defaulter  

The model helps banks make **data-driven approval decisions** and reduce default risk.

---

## Cloud Architecture
- Created Azure Resource Group  
- Created Azure Storage Account (ADLS Gen2)  
- Uploaded raw CSV files to Blob container  
- Provisioned Azure Databricks workspace & cluster  
- Connected Databricks to ADLS using **SAS token**  

---

## Dataset
Two CSV files:

- `application_record1.csv` → Demographic & application details  
- `credit_record1.csv` → Monthly credit history and status  

Both datasets are joined using common applicant **ID**.

---

## Tools & Technologies
- Azure Databricks  
- Azure Data Lake Storage Gen2  
- PySpark  
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost (SparkXGBClassifier)

---

##  Project Workflow
1. Data ingestion from ADLS Gen2  
2. Data cleaning and null handling  
3. Exploratory Data Analysis  
4. Feature engineering  
5. Categorical encoding using StringIndexer  
6. Feature vector assembly  
7. Train / Validation / Test split (70/15/15)  
8. Baseline model – Logistic Regression  
9. Final model – XGBoost  
10. Hyperparameter tuning  
11. Threshold optimization  
12. Model evaluation  

---

## Feature Engineering Highlights
- Derived **AGE** and **YEARS_EMPLOYED**  
- Created **MONTH_HISTORY_LENGTH**  
- Income binning using QuantileDiscretizer  
- Outlier capping for income and employment years  
- Converted credit STATUS into binary label  

---

## Models Used
| Model | Purpose |
|  -----| --------|
| Logistic Regression | Baseline |
| XGBoost Classifier | Final Model |

---

## Final Model Performance (Threshold = 0.75)

- AUC: **0.898**  
- Precision: **0.894**  
- Recall: **0.737**  
- F1 Score: **0.808**

Confusion Matrix:

| | Predicted Default | Predicted Safe |
|--|--|--|
| Actual Default | 42 | 5 |
| Actual Safe | 15 | 3553 |

---

## Business Impact
- Minimizes approving risky customers  
- Captures majority of defaulters  
- Threshold adjustable based on bank risk appetite  
- Supports scalable and consistent underwriting decisions  

---

## Visual Results

### Model Comparison (AUC)
XGBoost significantly outperforms Logistic Regression, showing superior ranking ability for risky applicants.

![Model Comparison](images/model_comparison_auc.png)

---

### Class Imbalance in Target Variable
The dataset is highly imbalanced, with defaulters representing a very small percentage of applicants — a realistic banking scenario.

![Class Distribution](images/class_distribution.png)

---

### Confusion Matrix (Threshold = 0.75)
Shows strong detection of defaulters and very low number of risky approvals.

![Confusion Matrix](images/confusion_matrix.png)

---

### ROC Curve (AUC = 0.898)
Demonstrates excellent separability between safe and risky applicants.

![ROC Curve](images/roc_curve.png)

---

### Precision–Recall Curve (AP = 0.749)
Highlights strong precision at high recall levels for the minority (defaulter) class.

![PR Curve](images/pr_curve.png)

---

### XGBoost Feature Importance
Most influential drivers of credit risk.

![Feature Importance](images/feature_importance.png)

---

## Project Structure
credit-card-approval-pyspark/
├── data/
├── notebooks/
├── src/
├── images/
├── README.md
└── requirements.txt

---
## Key Learnings
- Working with Spark on large datasets  
- Cloud-based ML pipeline on Azure  
- Handling imbalanced datasets  
- Business-oriented evaluation using precision, recall & AUC 