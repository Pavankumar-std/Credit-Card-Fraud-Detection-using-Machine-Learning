# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview

Credit card fraud detection is a highly imbalanced classification problem because fraudulent transactions occur much less frequently than genuine transactions.

This project develops a machine learning-based fraud detection system that identifies potentially fraudulent credit card transactions and compares multiple classification algorithms using appropriate evaluation metrics.

The project focuses not only on model accuracy but also on Precision, Recall, F1-score, ROC-AUC, PR-AUC, confusion matrices, and threshold analysis.

---

## 🎯 Problem Statement

Fraudulent transactions are rare compared with legitimate transactions, creating a highly imbalanced classification problem.

The objective of this project is to build and evaluate machine learning models that can detect fraudulent transactions while minimizing incorrect fraud alerts.

Because of the severe class imbalance, accuracy alone is not sufficient for evaluating the models.

---

## 📊 Dataset

The dataset contains:

- **284,807 total transactions**
- **284,315 genuine transactions**
- **492 fraudulent transactions**
- **30 input features**
- **1 target variable**

### Target Variable

The `Class` column represents the transaction type:

- `0` → Genuine transaction
- `1` → Fraudulent transaction

The dataset contains the following input features:

- `Time`
- `V1` to `V28`
- `Amount`

The `V1`–`V28` features are anonymized numerical features.

---

## ⚠️ Class Imbalance

The dataset is highly imbalanced:

| Class | Transactions | Percentage |
|---|---:|---:|
| Genuine | 284,315 | 99.83% |
| Fraudulent | 492 | 0.17% |

This severe imbalance makes accuracy an unreliable metric.

For example, a model could predict almost every transaction as genuine and still achieve very high accuracy while failing to detect fraud.

Therefore, this project emphasizes:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib

### Environment

- Jupyter Notebook / JupyterLab

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Duplicate Removal
   ↓
Class Imbalance Analysis
   ↓
Exploratory Data Analysis
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ├── Logistic Regression
   ├── Weighted Logistic Regression
   ├── Random Forest
   ├── Weighted Random Forest
   └── XGBoost
   ↓
Model Evaluation
   ├── Precision
   ├── Recall
   ├── F1-score
   ├── ROC-AUC
   ├── PR-AUC
   └── Confusion Matrix
   ↓
Threshold Analysis
   ↓
Feature Importance
   ↓
Model Comparison
   ↓
Final Model Selection
