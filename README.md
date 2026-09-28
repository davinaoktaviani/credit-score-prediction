# Credit Score Prediction

An end-to-end machine learning project that predicts customers' credit scores into three categories: **Poor, Standard, and Good**.

The project covers the complete machine learning workflow, from data preprocessing and model training to experiment tracking and application deployment. The trained model can be used through both a **local Streamlit application** and a **cloud-based deployment using AWS**.

---

## Project Overview

Credit scoring is an important process for understanding a customer's credit profile. However, analyzing multiple financial attributes manually can be challenging.

This project aims to build a machine learning model that classifies customers into three credit score categories:

* **Poor**
* **Standard**
* **Good**

The application allows users to enter customer financial information and receive a predicted credit score together with the probability of each class.

---

## Objectives

* Perform data preprocessing for credit-related data
* Build and compare multiple classification models
* Evaluate model performance using classification metrics
* Create a reusable preprocessing and modeling pipeline
* Track experiments using MLflow
* Deploy the model through a local Streamlit application
* Deploy the model using AWS services
* Provide real-time predictions through a Streamlit interface

---

## Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Preprocessing
     ↓
Train / Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
MLflow Experiment Tracking
     ↓
Best Model Selection
     ↓
Model Packaging
     ↓
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
Local Deployment      AWS Deployment
 │                       │
 ▼                       ▼
Streamlit             S3
                         ↓
                    SageMaker
                         ↓
                       EC2
                         ↓
                    Streamlit
```

---

## Models

Three classification algorithms were implemented and compared:

* **Logistic Regression**
* **Decision Tree**
* **Random Forest**

The preprocessing pipeline includes:

* Missing value imputation
* Numerical feature preprocessing
* Categorical feature encoding
* Feature transformation

---

## Model Performance

The models were evaluated using Accuracy, Precision, Recall, and F1-Score.

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   0.6530 |  0.653713 | 0.6530 | 0.650373 |
| Decision Tree       |   0.6388 |  0.639410 | 0.6388 | 0.639087 |
| Random Forest       |   0.7364 |  0.737963 | 0.7364 | 0.736878 |

### Best Performing Model

Random Forest achieved the highest performance among the evaluated models, with an accuracy of **73.64%** and an F1-score of **73.69%**.

---

## Experiment Tracking

**MLflow** was used to track model training experiments and evaluation results.

The experiment tracking process records:

* Model type
* Training metrics
* Test metrics
* Weighted F1-score
* Model artifacts

This enables systematic comparison between different machine learning models.

---

# Deployment

This project supports two deployment approaches:

## 1. Local Deployment

The local version runs the trained model and Streamlit application directly on a local machine.

```text
User
  ↓
Streamlit
  ↓
Preprocessing Pipeline
  ↓
Random Forest Model
  ↓
Prediction
```

The local application provides:

* Customer input form
* Credit score prediction
* Class probabilities
* Prediction results

---

## 2. AWS Deployment

The cloud version uses AWS services to host the machine learning inference workflow.

### Architecture

```text
                    ┌──────────────────┐
                    │   Trained Model  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Amazon S3     │
                    │   model.tar.gz   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Amazon SageMaker │
                    │ Inference Endpoint│
                    └────────┬─────────┘
                             │
                           Boto3
                             │
                             ▼
                    ┌──────────────────┐
                    │   Amazon EC2     │
                    │ Streamlit App    │
                    └────────┬─────────┘
                             │
                             ▼
                           User
```

### AWS Services

**Amazon S3**

Stores the packaged machine learning model before deployment.

**Amazon SageMaker**

Hosts the trained Scikit-learn model as an inference endpoint.

**Amazon EC2**

Hosts the Streamlit application that serves as the user interface.

**Boto3**

Connects the Streamlit application to the SageMaker Runtime API and sends prediction requests to the endpoint.

**IAM**

Provides the necessary permissions for AWS resources to communicate securely.

---

## AWS Configuration

The SageMaker deployment uses:

* Scikit-learn
* Framework version: `1.2-1`
* Instance type: `ml.m5.large`
* Region: `us-east-1`

The deployed endpoint returns:

* Predicted credit score
* Probability for each credit score category

---

## Streamlit Application

The Streamlit interface allows users to enter customer financial and credit information.

The application then displays:

* Predicted credit score
* Probability of each class
* Prediction results

Example:

```text
Credit Score: Standard

Poor       12.00%
Standard   71.00%
Good       17.00%
```

---

## Future Improvements

* Perform hyperparameter tuning to improve model performance.
* Add model explainability using SHAP.
* Improve the Streamlit user interface.
* Implement automated model retraining.
* Add monitoring for the deployed SageMaker endpoint.
* Automate cloud deployment using CI/CD.
