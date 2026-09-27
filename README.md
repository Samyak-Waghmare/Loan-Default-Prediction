# Loan Default Prediction

A machine learning pipeline to predict the probability of loan default using borrower and loan characteristics. Built and compared 3 classifiers — Logistic Regression, Random Forest, and XGBoost — on 255,347 loan records, with SMOTE for class imbalance and GridSearchCV for hyperparameter tuning.

---

## Results at a Glance

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 67.3% | 20.8% | **69.5%** | 32.1% | **74.3%** |
| Random Forest | 87.3% | 36.4% | 18.5% | 24.5% | 73.0% |
| XGBoost | 88.2% | 40.2% | 11.2% | 17.5% | 71.8% |

**Winner: Logistic Regression** — best AUC-ROC (74.3%) and highest recall (69.5%), correctly flagging 7 in 10 actual defaulters.

> XGBoost achieved 88.2% accuracy but only 11.2% recall — it missed nearly 9 out of 10 real defaulters, making it unsuitable for risk screening despite its high accuracy.

---

## Key Findings

- **Best model**: Logistic Regression (tuned) — AUC-ROC 74.3%, Recall 69.5%
- **Best hyperparameters**: `C=10, penalty=l1, solver=saga`
- **Class imbalance**: 11.6% default rate (23,722 defaulters in 255,347 records) — corrected with SMOTE, upsampling minority class to 180,555 samples
- **Top 5 risk drivers** (by coefficient magnitude): age, interest rate, employment duration, annual income, loan amount

---

## Project Structure

```
ML_LDP/
├── loan_default_prediction.ipynb   # Full ML pipeline (main notebook)
├── Loan_default.csv                # Dataset (255,347 records, 27 features)
├── loan_default_model.pkl          # Saved best model + scaler + feature list
├── model_comparison_results.csv    # Model comparison output
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Pipeline Overview

1. **Data Loading & EDA** — 255,347 records, 27 features engineered from 17 raw inputs
2. **Preprocessing** — median imputation, IQR outlier removal, one-hot encoding, StandardScaler
3. **Feature Engineering** — income-to-loan ratio, age groups, credit score bands, DTI categories
4. **Class Imbalance** — SMOTE oversampling (minority: 23,722 → 180,555 samples)
5. **Model Training** — Logistic Regression, Random Forest, XGBoost
6. **Evaluation** — Accuracy, Precision, Recall, F1, AUC-ROC; ROC curves + confusion matrices
7. **Tuning** — GridSearchCV with 5-fold cross-validation (60 fits)
8. **Model Saving** — best model persisted as `loan_default_model.pkl`

---

## Using the Saved Model

```python
import joblib
import pandas as pd

# Load saved model artifacts
artifacts = joblib.load('loan_default_model.pkl')
model    = artifacts['model']
scaler   = artifacts['scaler']
features = artifacts['feature_columns']

# Prepare input data (same columns as training)
new_data = pd.DataFrame({ ... })

# Predict
X_scaled = scaler.transform(new_data[features])
prob     = model.predict_proba(X_scaled)[:, 1]
pred     = model.predict(X_scaled)

print(f"Default Probability: {prob[0]:.4f}")
print(f"Prediction: {'Default' if pred[0] == 1 else 'No Default'}")
```

---

## Installation

```bash
pip install -r requirements.txt
jupyter notebook loan_default_prediction.ipynb
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| pandas, numpy | Data manipulation |
| scikit-learn | ML models, GridSearchCV, metrics |
| xgboost | Gradient boosting |
| imbalanced-learn | SMOTE oversampling |
| matplotlib, seaborn | Visualization |
| joblib | Model serialization |
