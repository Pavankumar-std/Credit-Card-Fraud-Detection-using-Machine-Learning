# Credit Card Fraud Detection

## Project Overview

This project develops a machine learning classification system to detect potentially fraudulent credit card transactions. The main challenge is severe class imbalance because fraudulent transactions are much rarer than genuine transactions.

## Dataset

- Original dataset: 284,807 transactions and 31 columns
- Rows after duplicate removal: 283,726
- Features: 30
- Target: `Class`
- Class 0: Genuine transaction
- Class 1: Fraudulent transaction
- Missing values: None
- Duplicate rows removed: 1,081

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost
- Jupyter Notebook

## Project Workflow

1. Load and inspect the dataset
2. Check data types and missing values
3. Identify and remove duplicate rows
4. Perform exploratory data analysis (EDA)
5. Analyze the severe class imbalance
6. Perform a stratified train-test split
7. Apply SMOTE only to the training data
8. Train Logistic Regression, Random Forest, and XGBoost
9. Evaluate using Precision, Recall, F1-score, ROC-AUC, and PR-AUC
10. Compare the models and select the final model
11. Visualize confusion matrices, ROC curves, and Precision-Recall curves

## Why SMOTE?

The fraud class is highly underrepresented compared with genuine transactions.

SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic minority-class examples from existing minority samples. This helps machine learning models learn patterns from the rare fraud class more effectively.

SMOTE was applied only to the training data. The test data remained untouched to avoid data leakage and ensure a fair evaluation.

## Models Evaluated

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression + SMOTE | 0.5181 | 0.8776 | 0.6515 | 0.9740 | 0.7183 |
| Random Forest + SMOTE | 0.8586 | 0.8673 | 0.8629 | 0.9643 | 0.8656 |
| XGBoost + SMOTE | 0.7265 | 0.8673 | 0.7907 | 0.9785 | 0.8731 |

## Final Model

**Random Forest + SMOTE** was selected as the final model because it achieved the highest F1-score of **0.8629** and the highest precision of **0.8586**, while maintaining a strong fraud recall of **0.8673**.

XGBoost achieved the highest ROC-AUC (**0.9785**) and PR-AUC (**0.8731**), making it a strong alternative. However, Random Forest provided the best balance between precision and recall for the selected final model.

## Evaluation Metrics

Accuracy was not used as the main decision metric because the dataset is highly imbalanced.

The project focuses on:

- **Precision** – How many transactions predicted as fraud were actually fraudulent.
- **Recall** – How many actual fraudulent transactions were successfully detected.
- **F1-Score** – Balance between precision and recall.
- **ROC-AUC** – Measures the model's ability to distinguish between genuine and fraudulent transactions.
- **PR-AUC** – Particularly useful for evaluating models on highly imbalanced datasets.

## Visualizations

The project includes the following visualizations:

- Fraud vs Genuine transaction distribution
- Transaction amount distribution
- Fraud transaction analysis over time
- Feature comparison
- Model performance comparison
- Confusion matrices
- Precision-Recall curves
- ROC curves

## Key Findings

- Credit card fraud detection is a highly imbalanced classification problem.
- SMOTE was used to address class imbalance in the training data.
- Logistic Regression achieved high fraud recall but produced more false-positive predictions.
- XGBoost achieved the highest ROC-AUC and PR-AUC.
- Random Forest + SMOTE achieved the highest precision and F1-score.
- Random Forest + SMOTE provided the best overall balance between precision and recall.

## Final Results

The selected Random Forest + SMOTE model achieved:

- **Precision:** 0.8586
- **Recall:** 0.8673
- **F1-Score:** 0.8629
- **ROC-AUC:** 0.9643
- **PR-AUC:** 0.8656

In the evaluated test set, the model correctly detected **85 out of 98 fraudulent transactions**, while producing **14 false-positive fraud alerts**.

## Conclusion

This project demonstrates how machine learning can be applied to credit card fraud detection in a highly imbalanced dataset.

The project used data preprocessing, exploratory data analysis, stratified train-test splitting, and SMOTE-based oversampling to improve learning from the minority fraud class.

Three machine learning models were compared: Logistic Regression, Random Forest, and XGBoost.

Based on the evaluation results, **Random Forest + SMOTE** was selected as the final model because it provided the best balance between precision, recall, and F1-score.

The project demonstrates the importance of handling class imbalance and using appropriate evaluation metrics instead of relying only on accuracy.

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── Credit_Card_Fraud_Detection.ipynb
├── creditcard.csv
├── credit_card_fraud_random_forest_smote.pkl
├── credit_card_fraud_random_forest.pkl
├── credit_card_fraud_scaler.pkl
├── credit_card_fraud_feature_names.pkl
└── README.md
