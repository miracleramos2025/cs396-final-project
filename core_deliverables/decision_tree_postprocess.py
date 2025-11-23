# decision_tree_postprocess.py
# Post-processing fairness: Equalized Odds ThresholdOptimizer

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)


#Load data

df = pd.read_csv("../lsac_data.csv")
if "ZFYGPA" in df.columns:
    df = df.rename(columns={"ZFYGPA": "zfygpa"})

needed = ["race", "gender", "lsat", "ugpa", "zfygpa"]
for c in needed:
    df[c] = pd.to_numeric(df[c], errors="coerce")


#Label: above median first-year GPA

median_cut = df["zfygpa"].median()
df["admit_sim"] = (df["zfygpa"] >= median_cut).astype(int)


#Features and protected attributes

X = (
    df.select_dtypes(include="number")
    .drop(columns=["zfygpa", "admit_sim", "race", "gender"], errors="ignore")
)

y = df["admit_sim"].astype(int)
gender = df["gender"].astype(int)
race = df["race"].astype(int)

X_train, X_test, y_train, y_test, g_train, g_test, r_train, r_test = train_test_split(
    X, y, gender, race, stratify=y, test_size=0.25, random_state=17
)

# ================================================================
# 4. Baseline decision tree (no reweighing here)
# ================================================================
# tree = DecisionTreeClassifier(
#     criterion="gini",
#     max_depth=4,
#     min_samples_leaf=50,
#     min_samples_split=50,
#     random_state=42,
# )

# tree.fit(X_train, y_train)

# proba_train = tree.predict_proba(X_train)[:, 1]
# proba_test = tree.predict_proba(X_test)[:, 1]


# Hyperparameter grid for decision tree
param_grid = {
    "max_depth": [2, 3, 4, 5, None],
    "min_samples_leaf": [20, 50, 100],
    "min_samples_split": [20, 50, 100],
}

base_tree = DecisionTreeClassifier(
    criterion="gini",
    random_state=42,
)

# GridSearchCV with 5-fold cross-validation, optimizing AUC
grid = GridSearchCV(
    estimator=base_tree,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
)

grid.fit(X_train, y_train)

print("Best params from CV:", grid.best_params_)
tree = grid.best_estimator_

proba = tree.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
proba_test = tree.predict_proba(X_test)[:, 1]

#Apply Equalized Odds post-processing
# We enforce equalized odds with respect to RACE
eo_optimizer = ThresholdOptimizer(
    estimator=tree,
    constraints="equalized_odds",
    predict_method="predict_proba"
)

eo_optimizer.fit(X_train, y_train, sensitive_features=r_train)

# Final predictions under Equalized Odds constraint
pred_post = eo_optimizer.predict(X_test, sensitive_features=r_test)


#  Evaluate performance and fairness

print(f"Postprocessed Model | Accuracy: {accuracy_score(y_test, pred_post):.3f} | AUC: {roc_auc_score(y_test, proba_test):.3f}")

def fairness(sens, name):
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=y_test,
        y_pred=pred_post,
        sensitive_features=sens
    )

    dp = demographic_parity_difference(y_test, pred_post, sensitive_features=sens)
    eo = equalized_odds_difference(y_test, pred_post, sensitive_features=sens)

    print(f"\n=== {name.upper()} ===")
    print("Selection rates:\n", mf.by_group["selection_rate"])
    print("Accuracies:\n", mf.by_group["accuracy"])
    print(f"DP difference: {dp:.3f}")
    print(f"EO difference: {eo:.3f}")

fairness(g_test, "gender")
fairness(r_test, "race")
