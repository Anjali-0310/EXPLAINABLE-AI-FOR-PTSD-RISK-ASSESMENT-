"""
PTSD Risk Assessment — Complete Training Pipeline
XAI Methods:
  1. SHAP  — Game-theory based exact attribution (TreeExplainer on XGBoost,
              KernelExplainer on Stacking Ensemble)
  2. LIME  — Local surrogate model, rule-based per-patient explanation
  3. Permutation Importance (ELI5-style) — Model-agnostic global importance
              directly on the Stacking Ensemble

Why all three?
  SHAP  → mathematically exact, gold standard attribution
  LIME  → human-readable IF-THEN rules for each patient
  Perm  → tests what the ENSEMBLE actually depends on globally

When all three agree on feature rankings → the finding is robust (XAI triangulation).
"""

import matplotlib
matplotlib.use("Agg")          # prevents tkinter thread crash

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── XAI imports ───────────────────────────────────────────────
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance

# ── sklearn imports ───────────────────────────────────────────
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report, precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import os
os.makedirs("plots", exist_ok=True)

def save(name):
    path = f"plots/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")

# ════════════════════════════════════════════════════════════════
# 1.  LOAD & FEATURES
# ════════════════════════════════════════════════════════════════
data = pd.read_csv("dataset/PTSD_research.csv")
data = data.fillna(0)

features = [
    "Age", "BMI", "psychological_resilience",
    "recent_accident", "family_accident",
    "witnessed_serious_injury", "witnessed_corpse",
    "traumatic_scene", "psychiatric_history"
]

X = data[features]
y = data["PTSD"]

print("\nCLASS DISTRIBUTION:")
print(y.value_counts())
print(f"Positive class ratio: {y.mean():.3f}")

# ── class distribution plot ───────────────────────────────────
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette=["#1B6CA8", "#C62828"])
plt.title("Class Distribution (Before SMOTE)")
plt.xlabel("PTSD (0 = No, 1 = Yes)")
plt.ylabel("Count")
save("01_class_distribution")

# ════════════════════════════════════════════════════════════════
# 2.  SPLIT  →  SMOTE (training only)  →  SCALE
# ════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain: {X_train.shape[0]}  Test: {X_test.shape[0]}")

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE (training only — no leakage):")
print(pd.Series(y_train_res).value_counts())

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled  = scaler.transform(X_test)

# ════════════════════════════════════════════════════════════════
# 3.  MODELS
# ════════════════════════════════════════════════════════════════
xgb_model = XGBClassifier(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_split=5,
    min_samples_leaf=2, class_weight="balanced",
    random_state=42, n_jobs=-1
)

stacking_model = StackingClassifier(
    estimators=[
        ("xgb", XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42
        )),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=1
        )),
    ],
    final_estimator=LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        random_state=42, solver="lbfgs"
    ),
    cv=5, stack_method="predict_proba", n_jobs=1
)

print("\nTraining XGBoost...")
xgb_model.fit(X_train_scaled, y_train_res)
print("Training Random Forest...")
rf_model.fit(X_train_scaled, y_train_res)
print("Training Stacking Ensemble...")
stacking_model.fit(X_train_scaled, y_train_res)

# ════════════════════════════════════════════════════════════════
# 4.  EVALUATION
# ════════════════════════════════════════════════════════════════
THRESHOLD = 0.4

def evaluate(name, mdl):
    prob = mdl.predict_proba(X_test_scaled)[:, 1]
    pred = (prob > THRESHOLD).astype(int)
    fpr, tpr, _ = roc_curve(y_test, prob)
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, pred),                   4),
        "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, pred),                     4),
        "F1":        round(f1_score(y_test, pred),                         4),
        "ROC-AUC":   round(auc(fpr, tpr),                                  4),
        "AP":        round(average_precision_score(y_test, prob),          4),
        "prob": prob, "pred": pred
    }

results = []
for n, m in [("XGBoost", xgb_model),
             ("Random Forest", rf_model),
             ("Stacking Ensemble", stacking_model)]:
    r = evaluate(n, m)
    results.append(r)
    print(f"\n--- {n} ---")
    for k in ["Accuracy","Precision","Recall","F1","ROC-AUC","AP"]:
        print(f"  {k:12s}: {r[k]}")

df_res = pd.DataFrame(results).drop(columns=["prob","pred"])
print("\n" + "="*65)
print("MODEL COMPARISON")
print("="*65)
print(df_res.to_string(index=False))

# best model by F1
best_name   = df_res.loc[df_res["F1"].idxmax(), "Model"]
best_result = next(r for r in results if r["Model"] == best_name)
prob_best   = best_result["prob"]
pred_best   = best_result["pred"]
print(f"\n✅ Best model by F1: {best_name}  (F1={best_result['F1']})")
print(classification_report(y_test, pred_best, zero_division=0))

# model comparison bar chart
metrics_to_plot = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
df_plot = df_res.set_index("Model")[metrics_to_plot]
fig, ax = plt.subplots(figsize=(12, 5))
df_plot.plot(kind="bar", ax=ax, width=0.6)
ax.set_title("Model Comparison — All Metrics")
ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", fontsize=7, padding=2)
ax.legend(loc="lower right")
plt.tight_layout()
save("02_model_comparison")

# confusion matrix
cm = confusion_matrix(y_test, pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Pred: No PTSD","Pred: PTSD"],
            yticklabels=["Actual: No PTSD","Actual: PTSD"])
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
save("03_confusion_matrix")

# ROC curves
plt.figure(figsize=(8, 6))
for r in results:
    fpr, tpr, _ = roc_curve(y_test, r["prob"])
    plt.plot(fpr, tpr, label=f"{r['Model']} (AUC={r['ROC-AUC']:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models")
plt.legend(loc="lower right"); plt.tight_layout()
save("04_roc_curves")

# precision-recall curves
plt.figure(figsize=(8, 6))
baseline = y_test.mean()
plt.axhline(baseline, color="gray", linestyle=":", label=f"Random baseline (AP={baseline:.2f})")
for r in results:
    prec, rec, _ = precision_recall_curve(y_test, r["prob"])
    plt.plot(rec, prec, label=f"{r['Model']} (AP={r['AP']:.2f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curves — All Models  (Primary Metric for Imbalanced Data)")
plt.legend(loc="upper right"); plt.tight_layout()
save("05_pr_curves")

# calibration curves
plt.figure(figsize=(8, 6))
for r in results:
    pt, pp = calibration_curve(y_test, r["prob"], n_bins=10)
    plt.plot(pp, pt, marker='o', label=r["Model"])
plt.plot([0,1],[0,1],'k--', label="Perfect")
plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction of Positives")
plt.title("Calibration Curves — All Models")
plt.legend(); plt.tight_layout()
save("06_calibration")

# threshold vs recall
thrs  = np.linspace(0, 1, 50)
recs  = [recall_score(y_test,(prob_best>t).astype(int), zero_division=0) for t in thrs]
plt.figure()
plt.plot(thrs, recs)
plt.axvline(THRESHOLD, color='r', linestyle='--', label=f"Threshold={THRESHOLD}")
plt.xlabel("Threshold"); plt.ylabel("Recall")
plt.title(f"Threshold vs Recall — {best_name}")
plt.legend(); plt.tight_layout()
save("07_threshold_recall")

# feature importance – XGBoost
plt.figure(figsize=(8, 5))
idx = np.argsort(xgb_model.feature_importances_)
sns.barplot(x=xgb_model.feature_importances_[idx], y=np.array(features)[idx], color="#1B6CA8")
plt.title("Feature Importance — XGBoost"); plt.tight_layout()
save("08_feat_imp_xgb")

# feature importance – RF
plt.figure(figsize=(8, 5))
idx = np.argsort(rf_model.feature_importances_)
sns.barplot(x=rf_model.feature_importances_[idx], y=np.array(features)[idx], color="#2E7D32")
plt.title("Feature Importance — Random Forest"); plt.tight_layout()
save("09_feat_imp_rf")

# ════════════════════════════════════════════════════════════════
# 5.  CROSS-VALIDATION
# ════════════════════════════════════════════════════════════════
print("\nCross-Validation — Stacking Ensemble (5-fold, pre-SMOTE data):")
cv_split   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_train_cv = scaler.transform(X_train)   # pre-SMOTE, no leakage

for metric in ["accuracy", "f1", "roc_auc"]:
    scores = cross_val_score(
        stacking_model, X_train_cv, y_train,
        cv=cv_split, scoring=metric, n_jobs=1
    )
    print(f"  {metric:10s}: {scores.mean():.4f} ± {scores.std():.4f}")

# ════════════════════════════════════════════════════════════════
# 6a.  XAI METHOD 1 — SHAP
#      Game-theory based.  Answers: "How much did each feature
#      contribute mathematically to THIS prediction?"
#
#      Two levels:
#        • TreeExplainer on XGBoost  → exact SHAP values (fast)
#        • KernelExplainer on Stacking Ensemble → approximate
#          (model-agnostic, explains the final decision maker)
# ════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("XAI METHOD 1 — SHAP")
print("="*65)

# ── 6a-i.  SHAP on XGBoost (exact, TreeExplainer) ────────────
print("\n[SHAP] TreeExplainer on XGBoost base model (exact values)...")
explainer_xgb   = shap.TreeExplainer(xgb_model)
shap_values_xgb = explainer_xgb(X_test_scaled)

# Global summary plot — all test patients
plt.figure()
shap.summary_plot(shap_values_xgb, X_test_scaled,
                  feature_names=features, show=False)
plt.title("SHAP Global Summary — XGBoost\n"
          "(Red = high feature value  |  Right = pushes toward PTSD)")
save("10_shap_xgb_global")

# Local waterfall — first test patient
plt.figure()
shap.plots.waterfall(shap_values_xgb[0], show=False)
plt.title("SHAP Local Waterfall — XGBoost (Patient #0)")
save("11_shap_xgb_waterfall")

# Print top features by mean |SHAP|
mean_abs_shap_xgb = pd.Series(
    np.abs(shap_values_xgb.values).mean(axis=0),
    index=features
).sort_values(ascending=False)
print("\nTop features by mean |SHAP| — XGBoost:")
print(mean_abs_shap_xgb.to_string())

# ── 6a-ii.  SHAP on Stacking Ensemble (KernelExplainer) ──────
print("\n[SHAP] KernelExplainer on Stacking Ensemble (approximate)...")
print("       (Uses k-means background from training data — no test leakage)")

np.random.seed(42)
background_km = shap.kmeans(X_train_scaled, 100)   # 100 cluster centres

def ensemble_predict_proba(X_arr):
    return stacking_model.predict_proba(X_arr)[:, 1]

explainer_ens = shap.KernelExplainer(ensemble_predict_proba, background_km)

# Explain 100 randomly sampled test patients (full test set is slow)
np.random.seed(42)
sample_idx          = np.random.choice(len(X_test_scaled), size=100, replace=False)
X_explain           = X_test_scaled[sample_idx]

np.random.seed(42)
shap_vals_ens       = explainer_ens.shap_values(X_explain, nsamples=2048)

# Global summary — ensemble
plt.figure()
shap.summary_plot(
    shap_vals_ens,
    pd.DataFrame(X_explain, columns=features),
    feature_names=features, show=False
)
plt.title("SHAP Global Summary — Stacking Ensemble Output\n"
          "(KernelExplainer — model-agnostic)")
save("12_shap_ensemble_global")

# Mean |SHAP| bar — ensemble
mean_abs_shap_ens  = np.abs(shap_vals_ens).mean(axis=0)
sorted_ens_idx     = np.argsort(mean_abs_shap_ens)[::-1]
plt.figure(figsize=(8, 5))
sns.barplot(
    x = mean_abs_shap_ens[sorted_ens_idx],
    y = np.array(features)[sorted_ens_idx],
    color = "#1B6CA8"
)
plt.title("SHAP Mean |Value| — Stacking Ensemble\n"
          "(Feature impact on PTSD risk — averaged over 100 patients)")
plt.xlabel("Mean |SHAP Value|")
plt.tight_layout()
save("13_shap_ensemble_bar")

# Local waterfall — ensemble (first explained patient)
exp_local = shap.Explanation(
    values       = shap_vals_ens[0],
    base_values  = explainer_ens.expected_value,
    data         = X_explain[0],
    feature_names= features
)
plt.figure()
shap.plots.waterfall(exp_local, show=False)
plt.title("SHAP Local Waterfall — Stacking Ensemble (Patient #0)")
save("14_shap_ensemble_waterfall")

print("\nTop features by mean |SHAP| — Stacking Ensemble:")
mean_ens_series = pd.Series(mean_abs_shap_ens, index=features).sort_values(ascending=False)
print(mean_ens_series.to_string())

# ════════════════════════════════════════════════════════════════
# 6b.  XAI METHOD 2 — LIME
#      Local surrogate model.
#      Answers: "What simple rule explains THIS ONE patient?"
#      Builds a local linear model by perturbing the input.
#      Model-agnostic — applied to the Stacking Ensemble.
# ════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("XAI METHOD 2 — LIME")
print("="*65)

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data         = X_train_scaled,
    feature_names         = features,
    class_names           = ["No PTSD", "PTSD"],
    mode                  = "classification",
    discretize_continuous = True,
    random_state          = 42
)

def ensemble_predict_for_lime(X_arr):
    """LIME needs predict_proba returning (n_samples, n_classes)."""
    return stacking_model.predict_proba(X_arr)

# ── Explain 3 representative patients ────────────────────────
for patient_idx in [0, 1, 2]:
    print(f"\n[LIME] Explaining test patient #{patient_idx}...")

    lime_exp = lime_explainer.explain_instance(
        data_row     = X_test_scaled[patient_idx],
        predict_fn   = ensemble_predict_for_lime,
        num_features = len(features),
        num_samples  = 3000,
        labels       = (1,)          # explain PTSD class
    )

    rules   = lime_exp.as_list(label=1)
    print(f"  LIME rules for patient #{patient_idx}:")
    for rule, weight in sorted(rules, key=lambda x: abs(x[1]), reverse=True):
        direction = "↑ PTSD" if weight > 0 else "↓ PTSD"
        print(f"    {direction}  {rule:45s}  weight={weight:+.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sorted_rules  = sorted(rules, key=lambda x: abs(x[1]), reverse=True)
    rule_labels   = [r[0] for r in sorted_rules]
    rule_weights  = [r[1] for r in sorted_rules]
    colors        = ["#C62828" if w > 0 else "#1B6CA8" for w in rule_weights]

    ax.barh(rule_labels[::-1], rule_weights[::-1],
            color=colors[::-1], edgecolor="none", height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME Weight  (positive = supports PTSD prediction)", fontsize=10)
    ax.set_title(
        f"LIME — Local Rule-Based Explanation\n"
        f"Stacking Ensemble  |  Test Patient #{patient_idx}",
        fontsize=10
    )
    ax.tick_params(axis="y", labelsize=8.5)
    ax.spines[["top","right"]].set_visible(False)
    red_p  = mpatches.Patch(color="#C62828", label="Supports PTSD ↑")
    blue_p = mpatches.Patch(color="#1B6CA8", label="Contradicts PTSD ↓")
    ax.legend(handles=[red_p, blue_p], fontsize=8)
    plt.tight_layout()
    save(f"15_lime_patient_{patient_idx}")

# ── LIME across multiple patients — aggregate top features ────
print("\n[LIME] Aggregating feature importance across 50 test patients...")
lime_agg = {f: [] for f in features}

for i in range(min(50, len(X_test_scaled))):
    exp = lime_explainer.explain_instance(
        data_row   = X_test_scaled[i],
        predict_fn = ensemble_predict_for_lime,
        num_features = len(features),
        num_samples  = 500,          # fewer samples for speed in aggregation
        labels = (1,)
    )
    for rule, weight in exp.as_list(label=1):
        for feat in features:
            if feat in rule:
                lime_agg[feat].append(abs(weight))
                break

lime_mean_abs = {f: np.mean(v) if v else 0.0 for f, v in lime_agg.items()}
lime_series   = pd.Series(lime_mean_abs).sort_values(ascending=False)

print("\nAggregated LIME mean |weight| across 50 patients:")
print(lime_series.to_string())

idx = np.argsort(list(lime_mean_abs.values()))
plt.figure(figsize=(8, 5))
sns.barplot(
    x = np.array(list(lime_mean_abs.values()))[idx],
    y = np.array(list(lime_mean_abs.keys()))[idx],
    color = "#E65100"
)
plt.title("LIME Aggregated Feature Importance\n"
          "(Mean |weight| across 50 patients — Stacking Ensemble)")
plt.xlabel("Mean |LIME Weight|")
plt.tight_layout()
save("16_lime_aggregated")

# ════════════════════════════════════════════════════════════════
# 6c.  XAI METHOD 3 — PERMUTATION IMPORTANCE (ELI5-style)
#      Statistical perturbation.
#      Answers: "What happens to model performance if I randomly
#      shuffle this feature?"
#      Directly on the Stacking Ensemble — fully model-agnostic.
#      No approximation — works on any black-box model.
# ════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("XAI METHOD 3 — PERMUTATION IMPORTANCE (ELI5-style)")
print("="*65)
print("Computing directly on Stacking Ensemble (model-agnostic)...")

perm_result = permutation_importance(
    estimator    = stacking_model,
    X            = X_test_scaled,
    y            = y_test,
    n_repeats    = 30,           # 30 shuffles per feature → stable estimate
    random_state = 42,
    scoring      = "average_precision",   # PR-AUC — primary metric
    n_jobs       = 1
)

perm_series = pd.Series(
    perm_result.importances_mean,
    index = features
).sort_values(ascending=False)

print("\nPermutation Importance — Stacking Ensemble")
print("(Mean drop in PR-AUC when feature is shuffled, over 30 repeats)")
print(f"{'Feature':35s}  {'Mean Drop':>10s}  {'Std':>8s}")
print("-" * 58)
for i, feat in enumerate(perm_series.index):
    mean = perm_result.importances_mean[features.index(feat)]
    std  = perm_result.importances_std[features.index(feat)]
    print(f"  {feat:33s}  {mean:>10.4f}  ±{std:.4f}")

# Sort by mean importance for plotting
sorted_feat_idx = np.argsort(perm_result.importances_mean)
sorted_feats    = np.array(features)[sorted_feat_idx]
sorted_means    = perm_result.importances_mean[sorted_feat_idx]
sorted_stds     = perm_result.importances_std[sorted_feat_idx]

# Bar chart with error bars
fig, ax = plt.subplots(figsize=(8, 5))
colors  = ["#C62828" if v > 0 else "#546E7A" for v in sorted_means]
ax.barh(sorted_feats, sorted_means, xerr=sorted_stds,
        color=colors, edgecolor="none", height=0.6,
        error_kw={"ecolor":"#333333","capsize":3,"linewidth":1.2})
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Mean Drop in PR-AUC  (higher = more important)", fontsize=10)
ax.set_title(
    "Permutation Importance (ELI5-style)\n"
    "Stacking Ensemble — Directly Model-Agnostic\n"
    "(30 repeats, scored on PR-AUC, error bars = std)",
    fontsize=10
)
ax.tick_params(axis="y", labelsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
save("17_permutation_importance")

# Box plot of importance distribution (shows stability)
fig, ax = plt.subplots(figsize=(8, 5))
importances_all = perm_result.importances     # shape (n_features, n_repeats)
ax.boxplot(
    [importances_all[i] for i in sorted_feat_idx],
    vert     = False,
    labels   = sorted_feats,
    patch_artist = True,
    boxprops = dict(facecolor="#DBEAFE", color="#1B6CA8"),
    medianprops  = dict(color="#C62828", linewidth=2),
    whiskerprops = dict(color="#546E7A"),
    capprops     = dict(color="#546E7A")
)
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Drop in PR-AUC per shuffle", fontsize=10)
ax.set_title(
    "Permutation Importance — Distribution Over 30 Repeats\n"
    "(Stacking Ensemble  |  wider box = less stable importance)",
    fontsize=10
)
plt.tight_layout()
save("18_permutation_boxplot")

# ════════════════════════════════════════════════════════════════
# 6d.  XAI TRIANGULATION — COMPARE ALL 3 METHODS
#      When all three methods agree on which features are most
#      important → the finding is robust (XAI triangulation).
# ════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("XAI TRIANGULATION — SHAP vs LIME vs PERMUTATION")
print("="*65)

# Normalise each method to 0-1 for visual comparison
def normalise(series):
    mn, mx = series.min(), series.max()
    if mx - mn == 0:
        return series * 0
    return (series - mn) / (mx - mn)

shap_series = mean_ens_series.reindex(features)     # from SHAP ensemble
lime_series = lime_series.reindex(features)
perm_series = perm_series.reindex(features)

shap_norm = normalise(shap_series)
lime_norm = normalise(lime_series)
perm_norm = normalise(perm_series)

# Rank each method
r_shap = shap_series.rank(ascending=False)
r_lime = lime_series.rank(ascending=False)
r_perm = perm_series.rank(ascending=False)
avg_rank = ((r_shap + r_lime + r_perm) / 3).sort_values()

print("\nConsensus ranking (lower = more consistently important):")
print(f"{'Feature':35s}  {'SHAP Rank':>9}  {'LIME Rank':>9}  {'Perm Rank':>9}  {'Avg Rank':>9}  Agreement")
print("-" * 90)
for feat in avg_rank.index:
    rs = int(r_shap[feat]); rl = int(r_lime[feat]); rp = int(r_perm[feat])
    ra = avg_rank[feat]
    max_diff = max(rs, rl, rp) - min(rs, rl, rp)
    agree = "✅ Agree" if max_diff <= 3 else "⚠  Differ"
    print(f"  {feat:33s}  {rs:>9}  {rl:>9}  {rp:>9}  {ra:>9.2f}  {agree}")

# Top features by consensus
top_feats = avg_rank.head(6).index.tolist()
x = np.arange(len(top_feats)); w = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w,  [shap_norm[f] for f in top_feats], w,
       label="SHAP (ensemble)",       color="#1B6CA8", alpha=0.9)
ax.bar(x,      [lime_norm[f] for f in top_feats], w,
       label="LIME (aggregated)",     color="#E65100", alpha=0.9)
ax.bar(x + w,  [perm_norm[f] for f in top_feats], w,
       label="Permutation (ELI5)",    color="#2E7D32", alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(top_feats, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Normalised Importance (0–1)", fontsize=10)
ax.set_title(
    "XAI Triangulation — SHAP vs LIME vs Permutation Importance\n"
    "(Top 6 features by consensus rank  |  normalised for visual comparison)",
    fontsize=10
)
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
save("19_xai_triangulation")

# ── What do differences between methods mean? ─────────────────
print("\n" + "="*65)
print("INTERPRETATION — WHY METHODS MAY GIVE DIFFERENT RANKINGS")
print("="*65)
print("""
SHAP  : Exact Shapley values — considers ALL possible feature orderings.
        Captures interaction effects between features.
        Value represents log-odds contribution.

LIME  : Local linear approximation around each patient.
        Builds a simple boundary — may miss global non-linearity.
        Values represent linear regression coefficients of the surrogate.

Perm  : Shuffles one feature at a time globally.
        Captures how much the MODEL DEPENDS on each feature overall.
        Does NOT capture interaction effects (unlike SHAP).
        Scale: drop in PR-AUC score.

When they AGREE → the feature is robustly important regardless of
  how you measure it. High confidence finding.

When they DIFFER → the feature may have:
  - Non-linear interaction effects (SHAP captures, Perm misses)
  - Local vs global importance difference (LIME local, Perm global)
  - Correlated features (SHAP distributes credit, Perm may double-count)
""")

# ════════════════════════════════════════════════════════════════
# 7.  SAVE ALL ARTIFACTS
# ════════════════════════════════════════════════════════════════
joblib.dump(stacking_model, "ensemble_stacking.pkl")
joblib.dump(xgb_model,      "model_xgb.pkl")
joblib.dump(rf_model,       "model_rf.pkl")
joblib.dump(scaler,         "scaler.pkl")
joblib.dump(features,       "features.pkl")

print("\n" + "="*65)
print("✅  PIPELINE COMPLETE")
print(f"   Best model       : {best_name}  (F1={best_result['F1']})")
print(f"   Threshold        : {THRESHOLD}")
print(f"   XAI methods      : SHAP (TreeExplainer + KernelExplainer)")
print(f"                      LIME (LimeTabularExplainer)")
print(f"                      Permutation Importance (sklearn)")
print(f"   Plots saved to   : ./plots/  ({20} files)")
print("="*65)