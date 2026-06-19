"""
CodeAlpha Internship - Task 1: Credit Scoring Model
Author: Adarsh Srivastava 
Description: Predict creditworthiness using Logistic Regression, Decision Tree, and Random Forest.
Dataset: Simulated financial dataset (replace with real dataset like German Credit / Kaggle Credit Risk)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. GENERATE / LOAD DATASET
# ──────────────────────────────────────────────
def generate_dataset(n=1000, seed=42):
    """Generate a synthetic credit dataset. Replace this with real data loading."""
    np.random.seed(seed)
    data = {
        "age":             np.random.randint(18, 70, n),
        "income":          np.random.randint(20000, 150000, n),
        "loan_amount":     np.random.randint(1000, 50000, n),
        "loan_tenure":     np.random.randint(1, 60, n),          # months
        "num_credit_cards": np.random.randint(0, 8, n),
        "num_loans":       np.random.randint(0, 5, n),
        "missed_payments": np.random.randint(0, 10, n),
        "debt_to_income":  np.round(np.random.uniform(0.05, 0.95, n), 2),
        "employment_years": np.random.randint(0, 30, n),
        "credit_score":    np.random.randint(300, 850, n),
    }
    df = pd.DataFrame(data)
    # Target: 1 = creditworthy, 0 = not creditworthy
    risk_score = (
    0.4*(df["credit_score"]/850)
    - 0.3*df["debt_to_income"]
    - 0.2*(df["missed_payments"]/10)
    + 0.1*(df["employment_years"]/30)
)

    df["creditworthy"] = (risk_score > 0.25).astype(int)
    return df


# ──────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ──────────────────────────────────────────────
def exploratory_analysis(df):
    print("=" * 55)
    print("  CREDIT SCORING MODEL — Exploratory Data Analysis")
    print("=" * 55)
    print(f"\nDataset shape : {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"\nClass distribution:\n{df['creditworthy'].value_counts()}")
    print(f"\nFeature statistics:\n{df.describe().T.round(2)}\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    df["creditworthy"].value_counts().plot(kind="bar", ax=axes[0],
        color=["#e74c3c", "#2ecc71"], edgecolor="black")
    axes[0].set_title("Class Distribution")
    axes[0].set_xlabel("Creditworthy"); axes[0].set_ylabel("Count")

    df.boxplot(column="credit_score", by="creditworthy", ax=axes[1])
    axes[1].set_title("Credit Score vs Creditworthy")
    axes[1].set_xlabel("Creditworthy"); axes[1].set_ylabel("Credit Score")
    plt.sca(axes[1]); plt.title("Credit Score vs Creditworthy")

    corr = df.corr()
    sns.heatmap(corr, ax=axes[2], cmap="coolwarm", annot=False, linewidths=0.5)
    axes[2].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=120)
    plt.show()
    print("EDA plots saved → eda_plots.png\n")


# ──────────────────────────────────────────────
# 3. PREPROCESSING
# ──────────────────────────────────────────────
def preprocess(df):
    X = df.drop("creditworthy", axis=1)
    y = df["creditworthy"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test, X.columns.tolist(), scaler


# ──────────────────────────────────────────────
# 4. TRAIN & EVALUATE MODELS
# ──────────────────────────────────────────────
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()

    print(f"\n{'─'*45}")
    print(f"  Model : {name}")
    print(f"{'─'*45}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  ROC-AUC       : {roc_auc:.4f}")
    print(f"  CV Accuracy   : {cv_score:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Not Creditworthy", "Creditworthy"]))

    return {
        "name": name, "model": model,
        "acc": acc, "roc_auc": roc_auc, "cv": cv_score,
        "y_pred": y_pred, "y_proba": y_proba
    }


def plot_results(results, y_test, feature_names):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Credit Scoring — Model Comparison", fontsize=14, fontweight="bold")

    # 1. Accuracy bar chart
    names = [r["name"] for r in results]
    accs  = [r["acc"] for r in results]
    aucs  = [r["roc_auc"] for r in results]
    x = np.arange(len(names))
    bars = axes[0].bar(x - 0.2, accs, 0.35, label="Accuracy", color="#3498db")
    bars2 = axes[0].bar(x + 0.2, aucs, 0.35, label="ROC-AUC",  color="#e67e22")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].set_ylim(0, 1.1); axes[0].set_title("Accuracy & ROC-AUC"); axes[0].legend()

    # 2. ROC curves
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        axes[1].plot(fpr, tpr, label=f"{r['name']} (AUC={r['roc_auc']:.2f})")
    axes[1].plot([0,1],[0,1],"k--"); axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curves"); axes[1].legend()

    # 3. Feature importance from Random Forest
    rf_result = next(r for r in results if "Forest" in r["name"])
    importances = rf_result["model"].feature_importances_
    idx = np.argsort(importances)[::-1]
    axes[2].barh([feature_names[i] for i in idx], importances[idx], color="#9b59b6")
    axes[2].set_title("Feature Importances (Random Forest)")
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=120)
    plt.show()
    print("Model comparison saved → model_comparison.png\n")


# ──────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────
def main():
    # Load / generate data
    df = generate_dataset(n=2000)
    exploratory_analysis(df)

    # Preprocess
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(df)

    # Define models
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Decision Tree",       DecisionTreeClassifier(max_depth=6, random_state=42)),
        ("Random Forest",       RandomForestClassifier(n_estimators=100, random_state=42)),
    ]

    # Train & evaluate
    results = []
    for name, model in models:
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(res)

    # Plot comparison
    plot_results(results, y_test, feature_names)

    # Best model summary
    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\n✅ Best Model: {best['name']} | ROC-AUC: {best['roc_auc']:.4f}")
    print("\nDone! Check eda_plots.png and model_comparison.png")


if __name__ == "__main__":
    main()