import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
from sklearn.model_selection import GridSearchCV
from visualization_utils import save_results_visualization, log_summary_row

# Load data
df = pd.read_csv("../lsac_data.csv")
if "ZFYGPA" in df.columns:
    df = df.rename(columns={"ZFYGPA": "zfygpa"})

needed = ["race", "gender", "lsat", "ugpa", "zfygpa"]

for c in needed:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Label: above median zfygpa
# cutoff_value = np.quantile(df["zfygpa"], 0.60)
# df["admit_sim"] = (df["zfygpa"] >= cutoff_value).astype(int)
cutoff_value = np.median(df["zfygpa"])
df["admit_sim"] = (df["zfygpa"] >= cutoff_value).astype(int)


# Features (drop label, zfygpa, protected attrs)
X = ( df.select_dtypes(include="number").drop(columns=["zfygpa", "admit_sim", "race", "gender"], errors="ignore")
)
y = df["admit_sim"].astype(int)
gender = df["gender"].astype(int)
race = df["race"].astype(int)

X_train, X_test, y_train, y_test, g_train, g_test, r_train, r_test = train_test_split(
    X, y, gender, race, stratify=y, test_size=0.25, random_state=42
)

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


#Performance metrics
overall_acc = accuracy_score(y_test, pred)
overall_auc = roc_auc_score(y_test, proba)
print(f"Decision Tree | Accuracy: {overall_acc:.3f} | AUC: {overall_auc:.3f}")


#Fairness metrices
def fairness(sens, name):
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=y_test,
        y_pred=pred,
        sensitive_features=sens
    )

    dp = demographic_parity_difference(y_test, pred, sensitive_features=sens)
    eo = equalized_odds_difference(y_test, pred, sensitive_features=sens)

    print(f"\n=== {name.upper()} ===")
    print("Selection rates:\n", mf.by_group["selection_rate"])
    print("Accuracies:\n", mf.by_group["accuracy"])
    print(f"DP difference: {dp:.3f}")
    print(f"EO difference: {eo:.3f}")
    return dp, eo

dp_gender, eo_gender = fairness(g_test, "gender")
dp_race, eo_race = fairness(r_test, "race")

# Log one-row summary for this model
log_summary_row(
    model_name="decision_tree_cv_baseline",
    overall_accuracy=overall_acc,
    overall_auc=overall_auc,
    dp_gender=dp_gender,
    eo_gender=eo_gender,
    dp_race=dp_race,
    eo_race=eo_race,
)

# Visualizations for baseline decision tree
save_results_visualization(
    model_name="decision_tree_cv_baseline",
    y_test=y_test,
    pred=pred,
    proba=proba,
    sensitive_features_dict={"gender": g_test, "race": r_test},
)
