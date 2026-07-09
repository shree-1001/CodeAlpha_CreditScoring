# 💳 Credit Scoring Model — CodeAlpha Task 1

Predict an individual's **creditworthiness** using past financial data with multiple classification algorithms.

---

## 🎯 Objective
Build a classification model to determine whether a person is creditworthy (`1`) or not (`0`) based on financial features.

---

## 🧠 Algorithms Used
| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Rule-based tree model |
| Random Forest | Ensemble of decision trees (best performer) |

---

## 📊 Key Features
- `credit_score`, `debt_to_income`, `missed_payments`
- `income`, `loan_amount`, `loan_tenure`
- `num_credit_cards`, `num_loans`, `employment_years`, `age`

---

## 📈 Metrics Evaluated
- **Accuracy**
- **Precision, Recall, F1-Score**
- **ROC-AUC Score**
- **5-Fold Cross Validation**

---

## 🚀 How to Run

```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn

# Run the model
python credit_scoring.py
```

---

## 📁 Output Files
| File | Description |
|---|---|
| `eda_plots.png` | Class distribution, boxplot, correlation heatmap |
| `model_comparison.png` | Accuracy/AUC bar chart, ROC curves, feature importance |

---

## 📦 Dataset
Using a **synthetic dataset** in this example. You can replace it with:
- [German Credit Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- [Give Me Some Credit (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit)

---

## 🏢 CodeAlpha Internship
> Task 1 of 3 | Machine Learning Domain
