# Intelligent Fraud Detection System for Fintech Applications

## Diploma Project (Astana IT University)

**Authors:** Tangbike Kural, Kassiyet Sarsenbek
**Program:** BDA – Big Data Analysis
**Supervisor:** Mira Rakhimzhanova, PhD, Assistant Professor
**Final Grade:** 94/100

---

## Overview

This project presents an intelligent fraud detection system for fintech applications based on Machine Learning, Deep Learning, Graph-Based Learning, and Big Data technologies.

The system is designed to detect fraudulent financial transactions in real time using a scalable streaming architecture and multiple predictive models.

---

## Problem Statement

The rapid growth of digital banking, online payments, and e-commerce has significantly increased the risk of financial fraud.

Traditional rule-based systems often fail to detect new and evolving fraud patterns. Therefore, intelligent data-driven solutions are required to improve fraud detection accuracy and reduce financial losses.

---

## Dataset

**European Credit Card Fraud Dataset**

* Total transactions: **284,807**
* Fraud transactions: **492**
* Fraud ratio: **0.172%**
* Binary classification:

  * **0 = Normal**
  * **1 = Fraud**

The dataset is highly imbalanced and represents real-world fraud detection challenges.

---

## Data Preprocessing

The following preprocessing steps were performed:

* Missing value verification
* Duplicate removal
* Amount normalization
* Stratified train/test split
* Feature engineering

### Engineered Features

Additional behavioral features were created:

* hour
* is_night
* amount_log
* velocity_1h
* time_gap
* rolling statistics
* transaction behavior indicators

As a result, the feature space increased from **30 to 42 features**.

---

## Implemented Models

### Random Forest

* Ensemble tree-based classifier
* Best ROC-AUC performance

### XGBoost

* Gradient boosting model
* Best balance between accuracy and speed

### TabTransformer

* Transformer-based architecture for tabular data
* Uses self-attention to learn feature relationships

### GNN Lite

* Graph-inspired fraud detection model
* Uses synthetic entity relationships

### Autoencoder

* Unsupervised anomaly detection model
* Uses reconstruction error

### Ensemble Model

* ROC-AUC weighted averaging of all models

---

## System Architecture

The platform combines Big Data streaming and AI technologies:

```text
Fintech Application
        │
        ▼
   Apache Kafka
        │
        ▼
Apache Spark Streaming
        │
        ▼
 Feature Processing
        │
        ▼
 ML/DL Models
(RF, XGB, Transformer,
 GNN Lite, Autoencoder)
        │
        ▼
 Ensemble Prediction
        │
        ▼
      FastAPI
        │
        ▼
    PostgreSQL
        │
        ▼
 Monitoring Dashboard
```

---

## Technology Stack

### Machine Learning & Deep Learning

* Scikit-Learn
* XGBoost
* TensorFlow
* PyTorch

### Big Data

* Apache Kafka
* Apache Spark Streaming

### Backend

* FastAPI

### Database

* PostgreSQL

### Deployment

* Docker

### Programming Language

* Python

---

## Experimental Results

| Model          | Precision | Recall    | F1-Score  | ROC-AUC   |
| -------------- | --------- | --------- | --------- | --------- |
| Random Forest  | 0.806     | 0.806     | 0.806     | **0.986** |
| XGBoost        | **0.882** | 0.837     | **0.859** | 0.976     |
| TabTransformer | 0.028     | **0.888** | 0.054     | 0.966     |
| GNN Lite       | 0.172     | 0.878     | 0.287     | 0.964     |
| Autoencoder    | -         | -         | -         | 0.954     |

### Key Findings

* Random Forest achieved the highest ROC-AUC (**0.986**).
* XGBoost achieved the best F1-score (**0.859**).
* XGBoost had the lowest latency (**52 ms**).
* Transformer achieved the highest Recall (**0.888**).
* Autoencoder successfully detected anomalous transaction behavior.
* Ensemble prediction improved prediction stability.

---

## Practical Impact

The developed system can be used by:

* Banks
* Fintech companies
* Payment gateways
* Digital wallet platforms

Potential applications include:

* Real-time transaction monitoring
* Fraud prevention
* Risk management
* Financial cybersecurity enhancement

---

## Future Work

Future improvements may include:

* Real graph transaction datasets
* Federated learning
* Drift detection
* Automatic model retraining
* Extended SHAP explainability
* Production-scale deployment

---

## Academic Achievement

This project was successfully defended as a Bachelor's Diploma Project at **Astana IT University** and received a final evaluation score of **94/100**.
