"""
Simple Classification Model Build Using Machine Learning
Dataset: Breast Cancer Wisconsin Diagnostic
Author: Dhinesh8023
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

def run_pipeline():
    print("=" * 70)
    print(" SIMPLE CLASSIFICATION MODEL BUILD USING MACHINE LEARNING ")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Load Dataset
    # -------------------------------------------------------------
    print("\n[STEP 1] Loading Dataset...")
    cancer_data = load_breast_cancer()
    X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
    y = pd.Series(cancer_data.target, name="target")

    print(f"Dataset Dimensions: {X.shape[0]} rows, {X.shape[1]} columns")
    print(f"Target Classes: {cancer_data.target_names} -> 0: {cancer_data.target_names[0]} (malignant), 1: {cancer_data.target_names[1]} (benign)")
    print("\nFirst 5 rows of features:")
    print(X.head())

    # -------------------------------------------------------------
    # 2. Exploratory Data Analysis & Cleaning
    # -------------------------------------------------------------
    print("\n[STEP 2] Exploratory Data Analysis & Cleaning...")
    missing_vals = X.isnull().sum().sum()
    print(f"Total Missing Values in Dataset: {missing_vals}")

    class_counts = y.value_counts()
    print("\nClass Distribution:")
    for cls_idx, count in class_counts.items():
        cls_name = cancer_data.target_names[cls_idx]
        pct = (count / len(y)) * 100
        print(f" - Class {cls_idx} ({cls_name}): {count} samples ({pct:.2f}%)")

    # -------------------------------------------------------------
    # 3. Train-Test Split & Feature Scaling
    # -------------------------------------------------------------
    print("\n[STEP 3] Splitting Data (80% Train, 20% Test) & Feature Scaling...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape:  {X_test.shape}")

    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling completed using StandardScaler.")

    # -------------------------------------------------------------
    # 4. Model Training: Logistic Regression
    # -------------------------------------------------------------
    print("\n[STEP 4] Training Model 1: Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    print("Logistic Regression trained successfully.")

    # -------------------------------------------------------------
    # 5. Model Training: Decision Tree Classifier
    # -------------------------------------------------------------
    print("\n[STEP 5] Training Model 2: Decision Tree Classifier...")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=4)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    print("Decision Tree Classifier trained successfully.")

    # Also evaluate DT with default (unpruned) to see if max_depth=4 or default is preferred
    dt_default = DecisionTreeClassifier(random_state=42)
    dt_default.fit(X_train, y_train)
    y_pred_dt_default = dt_default.predict(X_test)

    # -------------------------------------------------------------
    # 6. Evaluation Metrics Calculation
    # -------------------------------------------------------------
    print("\n[STEP 6] Calculating Evaluation Metrics...")

    # Logistic Regression metrics
    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr)
    rec_lr = recall_score(y_test, y_pred_lr)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    report_lr = classification_report(y_test, y_pred_lr, target_names=cancer_data.target_names)

    # Decision Tree (max_depth=4) metrics
    acc_dt = accuracy_score(y_test, y_pred_dt)
    prec_dt = precision_score(y_test, y_pred_dt)
    rec_dt = recall_score(y_test, y_pred_dt)
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    report_dt = classification_report(y_test, y_pred_dt, target_names=cancer_data.target_names)

    # Decision Tree (default) metrics
    acc_dt_def = accuracy_score(y_test, y_pred_dt_default)
    prec_dt_def = precision_score(y_test, y_pred_dt_default)
    rec_dt_def = recall_score(y_test, y_pred_dt_default)

    print("\n--- Model 1: Logistic Regression ---")
    print(f"Accuracy:  {acc_lr:.4f} ({acc_lr*100:.2f}%)")
    print(f"Precision: {prec_lr:.4f} ({prec_lr*100:.2f}%)")
    print(f"Recall:    {rec_lr:.4f} ({rec_lr*100:.2f}%)")
    print("Confusion Matrix:")
    print(cm_lr)
    print("\nClassification Report:\n", report_lr)

    print("\n--- Model 2: Decision Tree Classifier (max_depth=4) ---")
    print(f"Accuracy:  {acc_dt:.4f} ({acc_dt*100:.2f}%)")
    print(f"Precision: {prec_dt:.4f} ({prec_dt*100:.2f}%)")
    print(f"Recall:    {rec_dt:.4f} ({rec_dt*100:.2f}%)")
    print("Confusion Matrix:")
    print(cm_dt)
    print("\nClassification Report:\n", report_dt)

    print("\n--- Model 2 (Default unconstrained Decision Tree) ---")
    print(f"Accuracy:  {acc_dt_def:.4f} ({acc_dt_def*100:.2f}%)")
    print(f"Precision: {prec_dt_def:.4f} ({prec_dt_def*100:.2f}%)")
    print(f"Recall:    {rec_dt_def:.4f} ({rec_dt_def*100:.2f}%)")

    # -------------------------------------------------------------
    # 7. Comparison Summary Table
    # -------------------------------------------------------------
    comparison_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall"],
        "Logistic Regression": [f"{acc_lr*100:.2f}%", f"{prec_lr*100:.2f}%", f"{rec_lr*100:.2f}%"],
        "Decision Tree": [f"{acc_dt*100:.2f}%", f"{prec_dt*100:.2f}%", f"{rec_dt*100:.2f}%"]
    })
    print("\n--- PERFORMANCE COMPARISON TABLE ---")
    print(comparison_df.to_string(index=False))

    # -------------------------------------------------------------
    # 8. Visualizations
    # -------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Visualization 1: Logistic Regression Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_lr,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=cancer_data.target_names,
        yticklabels=cancer_data.target_names
    )
    plt.title("Confusion Matrix - Logistic Regression", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12, labelpad=8)
    plt.ylabel("True Label", fontsize=12, labelpad=8)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_logistic.png", dpi=300)
    plt.close()
    print("\nSaved results/confusion_matrix_logistic.png")

    # Visualization 2: Decision Tree Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_dt,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        xticklabels=cancer_data.target_names,
        yticklabels=cancer_data.target_names
    )
    plt.title("Confusion Matrix - Decision Tree Classifier", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12, labelpad=8)
    plt.ylabel("True Label", fontsize=12, labelpad=8)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_decision_tree.png", dpi=300)
    plt.close()
    print("Saved results/confusion_matrix_decision_tree.png")

    # Visualization 3: Model Comparison Chart (Accuracy, Precision, Recall)
    metrics_names = ["Accuracy", "Precision", "Recall"]
    lr_scores = [acc_lr * 100, prec_lr * 100, rec_lr * 100]
    dt_scores = [acc_dt * 100, prec_dt * 100, rec_dt * 100]

    x = np.arange(len(metrics_names))
    width = 0.35

    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(8, 6))

    rects1 = ax.bar(x - width/2, lr_scores, width, label="Logistic Regression", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, dt_scores, width, label="Decision Tree", color="#2a9d8f")

    ax.set_ylabel("Score (%)", fontsize=12, labelpad=8)
    ax.set_title("Model Performance Comparison (LR vs DT)", fontsize=14, pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.set_ylim(80, 105)
    ax.legend(frameon=True, loc="lower right")

    # Value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=300)
    plt.close()
    print("Saved results/model_comparison.png")

    print("\nExecution complete! All models trained, evaluated, and plots generated.")

if __name__ == "__main__":
    run_pipeline()
