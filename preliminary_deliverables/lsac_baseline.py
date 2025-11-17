# lsac_baseline.py
# baseline: logistic regression + basic fairness summaries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference

# load data
df = pd.read_csv("../lsac_data.csv")
if "ZFYGPA" in df.columns:
    df = df.rename(columns={"ZFYGPA": "zfygpa"})

# keep only needed columns numeric
need = ["race", "gender", "lsat", "ugpa", "zfygpa"]
for c in need:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# label: admit top 40% by zfygpa 
cut = np.quantile(df["zfygpa"], 0.60)
df["admit_sim"] = (df["zfygpa"] >= cut).astype(int)

# features (drop label, zfygpa, protected attrs) 
X = df.select_dtypes(include="number").drop(columns=["zfygpa", "admit_sim", "race", "gender"], errors="ignore")
y = df["admit_sim"].astype(int)
g = df["gender"].astype(int)
r = df["race"].astype(int)

X_tr, X_te, y_tr, y_te, g_tr, g_te, r_tr, r_te = train_test_split(
    X, y, g, r, stratify=y, test_size=0.25, random_state=17
)

# baseline model 
clf = LogisticRegression(max_iter=300)
clf.fit(X_tr, y_tr)
proba = clf.predict_proba(X_te)[:, 1]
pred = (proba >= 0.5).astype(int)

# overall metrics
print(f"overall accuracy: {accuracy_score(y_te, pred):.3f} | AUC: {roc_auc_score(y_te, proba):.3f}")

# fairness summaries 
def show_fairness(sens, name):
    mf = MetricFrame(
        metrics={"selection_rate": selection_rate, "accuracy": accuracy_score},
        y_true=y_te, y_pred=pred, sensitive_features=sens
    )
    dp = demographic_parity_difference(y_true=y_te, y_pred=pred, sensitive_features=sens)
    eo = equalized_odds_difference(y_true=y_te, y_pred=pred, sensitive_features=sens)
    print(f"\n=== {name} ===")
    print("by-group selection_rate:\n", mf.by_group["selection_rate"])
    print("by-group accuracy:\n", mf.by_group["accuracy"])
    print(f"demographic_parity_difference: {dp:.3f}")
    print(f"equalized_odds_difference: {eo:.3f}")

show_fairness(g_te, "gender")
show_fairness(r_te, "race")
