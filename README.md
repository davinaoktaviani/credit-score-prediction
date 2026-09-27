# Credit Score Prediction

A machine learning classification project that predicts customers' credit scores into three categories: **Poor, Standard, and Good**.

This project implements a complete machine learning workflow, including data preprocessing, model training, evaluation, experiment tracking, and deployment through a Streamlit application.

---

## Project Overview

Credit scoring is an important process for understanding a customer's credit profile. However, analyzing multiple financial attributes manually can be challenging.

This project aims to build a machine learning model that can classify a customer's credit score based on their financial and credit-related information.

The prediction consists of three classes:

- **Poor**
- **Standard**
- **Good**

---

## Objectives

The main objectives of this project are:

- Perform data preprocessing for credit-related data
- Build classification models for credit score prediction
- Compare different machine learning algorithms
- Evaluate model performance using classification metrics
- Create a reusable preprocessing and modeling pipeline
- Deploy the prediction model through a Streamlit application

---

## Model Performance

The models were evaluated using Accuracy, Precision, Recall, and F1-Score.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6530 | 0.653713 | 0.6530 | 0.650373 |
| Decision Tree | 0.6388 | 0.639410 | 0.6388 | 0.639087 |
| Random Forest | 0.7364 | 0.737963 | 0.7364 | 0.736878 |

### Best Performing Model

Based on the evaluation results, **Random Forest** achieved the highest performance across all four evaluation metrics, with an accuracy of **73.64%** and an F1-score of **73.69%**.

---

## Future Improvements

- Improve model performance through hyperparameter tuning.
- Add model explainability.
- Improve the Streamlit user interface.
- Deploy the application to a cloud platform.
