

# random_forest.py
# Baseline Random Forest (textbook-style) + CV + Fairness + Feature Importance

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)


#  Load and prepare data

df = pd.read_csv("../lsac_data.csv")
if "ZFYGPA" in df.columns:
    df = df.rename(columns={"ZFYGPA": "zfygpa"})

needed = ["race", "gender", "lsat", "ugpa", "zfygpa"]
for c in needed:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Label: above-median first-year GPA
median_cut = df["zfygpa"].median()
df["admit_sim"] = (df["zfygpa"] >= median_cut).astype(int)

# Features: numeric, drop label, zfygpa, and protected attributes
X = (
    df.select_dtypes(include="number")
      .drop(columns=["zfygpa", "admit_sim", "race", "gender"], errors="ignore")
)

y = df["admit_sim"].astype(int)
gender = df["gender"].astype(int)
race = df["race"].astype(int)

(
    X_train, X_test,
    y_train, y_test,
    g_train, g_test,
    r_train, r_test
) = train_test_split(
    X, y, gender, race,
    stratify=y,
    test_size=0.25,
    random_state=17
)


#  Random Forest + Grid Search
#    (inspired by HOML: n_estimators ~ 500, max_leaf_nodes, max_features="sqrt")


rf_base = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,           # use all cores like in the book
    bootstrap=True
)

param_grid = {
    # number of trees
    "n_estimators": [200, 500],
    # tree size controls
    "max_depth": [None, 5, 8],
    "max_leaf_nodes": [None, 16, 32],
    "min_samples_split": [2, 20],
    "min_samples_leaf": [1, 10, 20],
    
    "max_features": ["sqrt"],  
}

grid = GridSearchCV(
    rf_base,
    param_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best params from CV:", grid.best_params_)
rf_model = grid.best_estimator_

# Predictions
proba = rf_model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

#  Overall performance
print(f"Random Forest | Accuracy: {accuracy_score(y_test, pred):.3f} | AUC: {roc_auc_score(y_test, proba):.3f}")


# Feature importance (HOML-style)

print("\nFeature importances (Random Forest):")
for name, score in zip(X.columns, rf_model.feature_importances_):
    print(f"  {name}: {score:.3f}")


# Fairness evaluation (no interventions yet)

def fairness(sens, group_name):
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=y_test,
        y_pred=pred,
        sensitive_features=sens
    )

    dp = demographic_parity_difference(y_test, pred, sensitive_features=sens)
    eo = equalized_odds_difference(y_test, pred, sensitive_features=sens)

    print(f"\n=== {group_name.upper()} ===")
    print("Selection rates:\n", mf.by_group["selection_rate"])
    print("Accuracies:\n", mf.by_group["accuracy"])
    print(f"DP difference: {dp:.3f}")
    print(f"EO difference: {eo:.3f}")

fairness(g_test, "gender")
fairness(r_test, "race")

