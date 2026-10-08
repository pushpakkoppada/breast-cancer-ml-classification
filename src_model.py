"""
Week 4 Internship Project
Breast Cancer Classification using Machine Learning

Models:
1. Logistic Regression with StandardScaler
2. Random Forest Classifier

The dataset is loaded from scikit-learn's built-in copy of the
Wisconsin Diagnostic Breast Cancer dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    roc_auc_score, classification_report
)

# Load public dataset
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print("Accuracy :", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall   :", recall_score(y_test, pred))
    print("F1       :", f1_score(y_test, pred))
    print("ROC-AUC  :", roc_auc_score(y_test, prob))
    print(classification_report(y_test, pred, target_names=data.target_names))

# Confusion matrix for the selected model can be plotted here.
