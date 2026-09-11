Credit Card Fraud Detection

Project Overview

This project develops a machine learning classification system to detect potentially fraudulent credit card transactions. The main challenge is severe class imbalance because fraudulent transactions are much rarer than genuine transactions.

Dataset

Rows after duplicate removal: 283,726

Features: 30

Target: Class

Class 0: Genuine transaction

Class 1: Fraudulent transaction

Missing values: None

Duplicate rows removed: 1,081

Technologies

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

imbalanced-learn (SMOTE)

XGBoost

Jupyter Notebook

Project Workflow

Load and inspect the dataset

Check data types and missing values

Identify and remove duplicate rows

Perform exploratory data analysis

Analyze the severe class imbalance

Perform a stratified train-test split

Apply SMOTE only to the training data

Train Logistic Regression, Random Forest, and XGBoost

Evaluate using Precision, Recall, F1-score, ROC-AUC, and PR-AUC

Compare the models and select the final model

Visualize confusion matrices, ROC curves, and Precision-Recall curves

Why SMOTE?

The fraud class is highly underrepresented. SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic minority-class training examples to help the models learn fraud patterns more effectively.

SMOTE was applied only to the training data. The test data remained untouched to avoid data leakage.

Models Evaluated

Model

Precision

Recall

F1-Score

ROC-AUC

PR-AUC

Logistic Regression + SMOTE

0.5181

0.8776

0.6515

0.9740

0.7183

Random Forest + SMOTE

0.8586

0.8673

0.8629

0.9643

0.8656

XGBoost + SMOTE

0.7265

0.8673

0.7907

0.9785

0.8731

Final Model

Random Forest + SMOTE was selected as the final model because it achieved the highest F1-score (0.8629) and highest precision (0.8586), while maintaining a strong fraud recall of 0.8673.

XGBoost achieved the highest ROC-AUC (0.9785) and PR-AUC (0.8731), so it was also a strong alternative. However, Random Forest provided the best balance between precision and recall for the selected final model.

Evaluation

Accuracy was not used as the main decision metric because the dataset is highly imbalanced. Precision, Recall, F1-score, ROC-AUC, and especially PR-AUC provide more useful information for fraud detection.

Key Findings

Class imbalance is a major challenge in fraud detection.

SMOTE improved the models' ability to identify minority-class fraud cases.

Logistic Regression achieved high recall but produced more false positives.

XGBoost showed excellent ranking performance.

Random Forest + SMOTE provided the best overall precision/F1 balance.

Conclusion

The project demonstrates how machine learning and SMOTE can be used to identify rare fraudulent transactions. Comparing multiple models with appropriate imbalance-aware metrics provides a more meaningful evaluation than relying on accuracy alone.
