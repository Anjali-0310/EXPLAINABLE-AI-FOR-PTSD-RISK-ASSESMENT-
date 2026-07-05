import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, classification_report
)

from xgboost import XGBClassifier
import joblib

# ================= LOAD =================
data = pd.read_csv("dataset/PTSD.csv")

# ================= TARGET =================
threshold = data["cbclptsd"].median()
data["PTSD"] = (data["cbclptsd"] >= threshold).astype(int)

# Drop original score (avoid leakage)
data = data.drop(columns=["cbclptsd"])

# ================= HANDLE MISSING =================
data = data.fillna(data.median(numeric_only=True))

# ================= SELECT NUMERIC FEATURES =================
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove("PTSD")

X_full = data[numeric_cols]
y = data["PTSD"]

print("\nCLASS DISTRIBUTION:")
print(y.value_counts())

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ================= SCALE =================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================= HANDLE IMBALANCE =================
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# ================= TEMP MODEL (FEATURE IMPORTANCE) =================
temp_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

temp_model.fit(X_train_scaled, y_train)

# ================= FEATURE IMPORTANCE =================
importance = temp_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": numeric_cols,
    "importance": importance
}).sort_values(by="importance", ascending=False)

# Select top 20 features
selected_features = feature_importance_df.head(20)["feature"].tolist()

print("\nTop Selected Features:")
print(selected_features)

# ================= REDUCE DATA =================
X = data[selected_features]

# Re-split with selected features
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Re-scale
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================= FINAL MODEL =================
model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ================= PREDICT =================
prob = model.predict_proba(X_test_scaled)[:, 1]

# Threshold tuning
threshold = 0.6
pred = (prob > threshold).astype(int)

# ================= METRICS =================
print("\nMODEL PERFORMANCE:")
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1:", f1_score(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ================= ROC =================
fpr, tpr, _ = roc_curve(y_test, prob)
plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0,1],[0,1], linestyle="--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# ================= FEATURE IMPORTANCE PLOT =================
plt.figure(figsize=(8,6))
sns.barplot(
    x="importance",
    y="feature",
    data=feature_importance_df.head(10)
)
plt.title("Top 10 Important Features")
plt.show()

# ================= SAVE =================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selected_features, "features.pkl")

print("\n✅ MODEL READY")