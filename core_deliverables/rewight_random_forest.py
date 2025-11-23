# random_forest.py
# Random Forest baseline (tuned) + race reweighting + fairness + feature importance

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

# Features: numeric, drop label, zfygpa, and protected attrs
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


# 2. Reweighting for race (Kamiran & Calders style)
#    We compute weights on TRAINING data only.

r_train_arr = np.array(r_train)
y_train_arr = np.array(y_train)

N = len(y_train_arr)
groups = np.unique(r_train_arr)   # race values
labels = np.unique(y_train_arr)   # 0, 1

# counts
N_g = {g: np.sum(r_train_arr == g) for g in groups}
N_y = {lab: np.sum(y_train_arr == lab) for lab in labels}
N_gy = {
    (g, lab): np.sum((r_train_arr == g) & (y_train_arr == lab))
    for g in groups for lab in labels
}

weights = np.zeros_like(y_train_arr, dtype=float)

for g in groups:
    for lab in labels:
        if N_gy[(g, lab)] == 0:
            continue
        # w(g, y) = (P(g) * P(y)) / P(g, y)
        w_gy = (N_g[g] * N_y[lab]) / (N * N_gy[(g, lab)])
        mask = (r_train_arr == g) & (y_train_arr == lab)
        weights[mask] = w_gy

weights[weights == 0] = 1.0  # safety fallback


#Random Forest + Grid Search (unweighted)
#We tune hyperparams on accuracy/AUC as before.

rf_base = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    bootstrap=True
)

param_grid = {
    "n_estimators": [200, 500],
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

# Now refit a fresh RF with those params BUT using the race weights
rf_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    bootstrap=True,
    **grid.best_params_
)

rf_model.fit(X_train, y_train, sample_weight=weights)

# Predictions on test set
proba = rf_model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)


# Overall performance

print(f"Random Forest (reweighted) | Accuracy: {accuracy_score(y_test, pred):.3f} | AUC: {roc_auc_score(y_test, proba):.3f}")


# Feature importance
print("\nFeature importances (Random Forest, reweighted):")
for name, score in zip(X.columns, rf_model.feature_importances_):
    print(f"  {name}: {score:.3f}")


# Fairness evaluation

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
