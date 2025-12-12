import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
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

#Reweighting for race (Kamiran & Calders, 2012)
#convert to numpy arrays for easier manipulation
# Convert to numpy arrays for easier indexing
r_train_arr = np.array(r_train)
y_train_arr = np.array(y_train)

N = len(y_train_arr)
groups = np.unique(r_train_arr)   # race values
labels = np.unique(y_train_arr)   # 0, 1

# counts by group and label
# N_g: count for each group
N_g = {g: np.sum(r_train_arr == g) for g in groups}
# N_y: count for each label
N_y = {lab: np.sum(y_train_arr == lab) for lab in labels}
# N_gy: count for each (group, label) combo
N_gy = {
    (g, lab): np.sum((r_train_arr == g) & (y_train_arr == lab))
    for g in groups for lab in labels
}

# compute weights w(g, y) = (P(g) * P(y)) / P(g, y)
# where P(g) = N_g / N, P(y) = N_y / N, P(g,y) = N_gy / N
weights = np.zeros_like(y_train_arr, dtype=float)

for g in groups:
    for lab in labels:
        if N_gy[(g, lab)] == 0:
            # avoid division by zero; if no such combo, skip
            continue

        w_gy = (N_g[g] * N_y[lab]) / (N * N_gy[(g, lab)])

        # assign this weight to all samples with group g and label lab
        mask = (r_train_arr == g) & (y_train_arr == lab)
        weights[mask] = w_gy

# just in case, if any weights are still zero (rare), set them to 1
weights[weights == 0] = 1.0


# Regularized Decision Tree model
tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=50,
    min_samples_split=50,
    random_state=42,
)

tree.fit(X_train, y_train, sample_weight=weights)
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
    model_name="decision_tree_reweighted",
    overall_accuracy=overall_acc,
    overall_auc=overall_auc,
    dp_gender=dp_gender,
    eo_gender=eo_gender,
    dp_race=dp_race,
    eo_race=eo_race,
)

# Visualizations (ROC, confusion matrix, fairness by group)
save_results_visualization(
    model_name="decision_tree_reweighted",
    y_test=y_test,
    pred=pred,
    proba=proba,
    sensitive_features_dict={"gender": g_test, "race": r_test},
)
