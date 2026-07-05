"""
PTSD Risk Assessment — Flask App
Fixed: XAI (SHAP + LIME + Permutation) now correctly
       computed and passed to predict.html template.
"""

from flask import Flask, render_template, request, redirect, session
import pandas as pd
import numpy as np
import joblib
import sqlite3
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from functools import wraps

# ── XAI imports ──────────────────────────────────────────────
import shap
import lime
import lime.lime_tabular

import matplotlib
matplotlib.use("Agg")   # MUST be before pyplot — prevents tkinter crash
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64

app = Flask(__name__)
app.secret_key = "super_secret_key"

# ════════════════════════════════════════════════════════════
# LOAD MODEL ARTIFACTS
# ════════════════════════════════════════════════════════════
model    = joblib.load("model.pkl")
scaler   = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# XGBoost base model — needed for exact SHAP TreeExplainer
# If train_model.py saved model_xgb.pkl use it, else fall back
try:
    xgb_model = joblib.load("model_xgb.pkl")
    print("✅ model_xgb.pkl loaded — SHAP TreeExplainer (exact) available")
except FileNotFoundError:
    xgb_model = None
    print("⚠  model_xgb.pkl not found — SHAP will use KernelExplainer on model.pkl")

# ── Build LIME explainer ONCE at startup (not per request) ───
# Reads training data to build the background distribution
try:
    _train_raw    = pd.read_csv("dataset/PTSD_research.csv")[features].fillna(0)
    _train_scaled = scaler.transform(_train_raw)

    LIME_EXPLAINER = lime.lime_tabular.LimeTabularExplainer(
        training_data         = _train_scaled,
        feature_names         = features,
        class_names           = ["No PTSD", "PTSD"],
        mode                  = "classification",
        discretize_continuous = True,
        random_state          = 42
    )

    # SHAP background — k-means from training data (correct, no test leakage)
    SHAP_BACKGROUND = shap.kmeans(_train_scaled, 50)

    print("✅ LIME explainer and SHAP background built from training data")

except Exception as e:
    LIME_EXPLAINER  = None
    SHAP_BACKGROUND = None
    print(f"⚠  XAI setup warning: {e}")
    print("   XAI charts will be unavailable. Check dataset/PTSD_research.csv path.")


# ════════════════════════════════════════════════════════════
# HELPER — matplotlib figure → base64 PNG string
# Embeds chart directly in HTML — no file saving needed
# ════════════════════════════════════════════════════════════
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


# ════════════════════════════════════════════════════════════
# XAI METHOD 1 — SHAP
# Game-theory based mathematical attribution
# Answers: "How much did each feature contribute to THIS score?"
# ════════════════════════════════════════════════════════════
def run_shap(scaled_input):
    """
    Returns (shap_data list, plot_b64 string, method string)
    shap_data = [{feature, value, effect, bar_size, bar_pct}, ...]
    """
    try:
        if xgb_model is not None:
            # Exact SHAP — TreeExplainer works directly on XGBoost
            explainer = shap.TreeExplainer(xgb_model)
            shap_exp  = explainer(scaled_input)
            shap_vals = shap_exp.values[0]          # shape (n_features,)
            method    = "TreeExplainer on XGBoost base model (exact)"
        else:
            # Approximate SHAP — KernelExplainer works on any model
            if SHAP_BACKGROUND is None:
                return [], None, "SHAP unavailable — training data not found"
            def _pred(X):
                return model.predict_proba(X)[:, 1]
            explainer = shap.KernelExplainer(_pred, SHAP_BACKGROUND)
            shap_vals = explainer.shap_values(scaled_input, nsamples=512)[0]
            method    = "KernelExplainer on model.pkl (approximate)"

        # Sort by absolute importance — most impactful first
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
        top_feats  = [features[i] for i in sorted_idx]
        top_vals   = [float(shap_vals[i]) for i in sorted_idx]

        # Build horizontal bar chart
        colors = ["#C62828" if v > 0 else "#1B6CA8" for v in top_vals]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(top_feats[::-1], top_vals[::-1],
                color=colors[::-1], edgecolor="none", height=0.55)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(
            "SHAP Value  (red = increases PTSD risk  |  blue = decreases PTSD risk)",
            fontsize=9
        )
        ax.set_title(
            f"SHAP — Feature Contributions per Patient\n({method})",
            fontsize=10
        )
        ax.tick_params(axis="y", labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=[
            mpatches.Patch(color="#C62828", label="Pushes risk UP ↑"),
            mpatches.Patch(color="#1B6CA8", label="Pushes risk DOWN ↓")
        ], fontsize=8)
        plt.tight_layout()
        plot_b64 = fig_to_base64(fig)

        # Structured data for template
        max_abs   = max(abs(v) for v in top_vals) or 1
        shap_data = []
        for feat, val in zip(top_feats, top_vals):
            bar_pct  = int(abs(val) / max_abs * 100)
            bar_size = (
                "bar-xl" if bar_pct > 75 else
                "bar-lg" if bar_pct > 50 else
                "bar-md" if bar_pct > 25 else "bar-sm"
            )
            shap_data.append({
                "feature":  feat,
                "value":    round(val, 4),
                "effect":   "increases risk" if val > 0 else "decreases risk",
                "bar_size": bar_size,
                "bar_pct":  bar_pct,
            })
        return shap_data, plot_b64, method

    except Exception as e:
        print(f"SHAP error: {e}")
        return [], None, f"SHAP error: {e}"


# ════════════════════════════════════════════════════════════
# XAI METHOD 2 — LIME
# Local surrogate model
# Answers: "What simple IF-THEN rule explains THIS patient?"
# ════════════════════════════════════════════════════════════
def run_lime(scaled_input):
    """
    Returns (lime_data list, plot_b64 string, method string)
    lime_data = [{rule, weight, effect, bar_size, bar_pct}, ...]
    """
    try:
        if LIME_EXPLAINER is None:
            return [], None, "LIME unavailable — training data not loaded"

        def _pred(X):
            return model.predict_proba(X)   # shape (n, 2) required by LIME

        exp = LIME_EXPLAINER.explain_instance(
            data_row     = scaled_input[0],
            predict_fn   = _pred,
            num_features = len(features),
            num_samples  = 2000,
            labels       = (1,)              # explain PTSD class (class index 1)
        )

        rules        = exp.as_list(label=1)
        rules_sorted = sorted(rules, key=lambda x: abs(x[1]), reverse=True)

        # Build chart
        r_labels  = [r[0] for r in rules_sorted]
        r_weights = [r[1] for r in rules_sorted]
        colors    = ["#C62828" if w > 0 else "#1B6CA8" for w in r_weights]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(r_labels[::-1], r_weights[::-1],
                color=colors[::-1], edgecolor="none", height=0.55)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(
            "LIME Weight  (positive = supports PTSD  |  negative = contradicts PTSD)",
            fontsize=9
        )
        ax.set_title(
            "LIME — Local Rule-Based Explanation per Patient\n"
            "(Local surrogate model on model.pkl)",
            fontsize=10
        )
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=[
            mpatches.Patch(color="#C62828", label="Supports PTSD prediction ↑"),
            mpatches.Patch(color="#1B6CA8", label="Contradicts PTSD prediction ↓")
        ], fontsize=8)
        plt.tight_layout()
        plot_b64 = fig_to_base64(fig)

        max_abs   = max(abs(w) for _, w in rules_sorted) or 1
        lime_data = []
        for rule, weight in rules_sorted:
            bar_pct  = int(abs(weight) / max_abs * 100)
            bar_size = (
                "bar-xl" if bar_pct > 75 else
                "bar-lg" if bar_pct > 50 else
                "bar-md" if bar_pct > 25 else "bar-sm"
            )
            lime_data.append({
                "rule":     rule,
                "weight":   round(float(weight), 4),
                "effect":   "increases risk" if weight > 0 else "decreases risk",
                "bar_size": bar_size,
                "bar_pct":  bar_pct,
            })
        return lime_data, plot_b64, "LIME local surrogate on model.pkl"

    except Exception as e:
        print(f"LIME error: {e}")
        return [], None, f"LIME error: {e}"


# ════════════════════════════════════════════════════════════
# XAI METHOD 3 — PERMUTATION IMPORTANCE (ELI5-style)
# Model-agnostic statistical perturbation
# Answers: "How much does risk change if this feature is removed?"
# Works directly on model.pkl — fully model-agnostic
# ════════════════════════════════════════════════════════════
def run_permutation(scaled_input):
    """
    Local variant: zero out each feature one by one,
    measure change in PTSD probability.
    Positive delta = feature was ADDING risk (risk drops when removed).
    Negative delta = feature was REDUCING risk (risk rises when removed).

    Returns (perm_data list, plot_b64 string, method string)
    perm_data = [{feature, delta, effect, bar_size, bar_pct}, ...]
    """
    try:
        base_prob = float(model.predict_proba(scaled_input)[0][1])
        impacts   = {}

        for i, feat in enumerate(features):
            perturbed       = scaled_input.copy()
            perturbed[0, i] = 0.0           # zero out this feature
            new_prob        = float(model.predict_proba(perturbed)[0][1])
            impacts[feat]   = round(base_prob - new_prob, 4)

        sorted_impacts = sorted(
            impacts.items(), key=lambda x: abs(x[1]), reverse=True
        )

        # Build chart
        feat_names = [x[0] for x in sorted_impacts]
        imp_vals   = [x[1] for x in sorted_impacts]
        colors     = ["#C62828" if v > 0 else "#1B6CA8" for v in imp_vals]

        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(feat_names[::-1], imp_vals[::-1],
                color=colors[::-1], edgecolor="none", height=0.55)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(
            "Δ PTSD Probability when feature is removed  "
            "(red = was adding risk  |  blue = was reducing risk)",
            fontsize=9
        )
        ax.set_title(
            "Permutation Importance (ELI5-style)\n"
            "Direct on model.pkl — Fully Model-Agnostic",
            fontsize=10
        )
        ax.tick_params(axis="y", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=[
            mpatches.Patch(color="#C62828", label="Was increasing PTSD risk"),
            mpatches.Patch(color="#1B6CA8", label="Was reducing PTSD risk")
        ], fontsize=8)
        plt.tight_layout()
        plot_b64 = fig_to_base64(fig)

        max_abs   = max(abs(v) for _, v in sorted_impacts) or 1
        perm_data = []
        for feat, delta in sorted_impacts:
            bar_pct  = int(abs(delta) / max_abs * 100)
            bar_size = (
                "bar-xl" if bar_pct > 75 else
                "bar-lg" if bar_pct > 50 else
                "bar-md" if bar_pct > 25 else "bar-sm"
            )
            perm_data.append({
                "feature":  feat,
                "delta":    round(delta, 4),
                "effect":   "increases risk" if delta > 0 else "decreases risk",
                "bar_size": bar_size,
                "bar_pct":  bar_pct,
            })
        return perm_data, plot_b64, "Permutation (local) — directly on model.pkl"

    except Exception as e:
        print(f"Permutation error: {e}")
        return [], None, f"Permutation error: {e}"


# ════════════════════════════════════════════════════════════
# DATABASE
# ════════════════════════════════════════════════════════════
def get_db():
    return sqlite3.connect("database.db")

def init_db():
    conn = get_db()
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT,
            email    TEXT UNIQUE,
            password TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER,
            probability REAL,
            result      TEXT,
            date        TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


# ════════════════════════════════════════════════════════════
# AUTH DECORATOR
# ════════════════════════════════════════════════════════════
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return wrapper


# ════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════

@app.route("/")
def home():
    if "user_id" in session:
        return redirect("/dashboard")
    return redirect("/login")


# ── LOGIN ────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email")
        password = request.form.get("password")

        conn = get_db()
        c    = conn.cursor()
        c.execute(
            "SELECT * FROM users WHERE email=? AND password=?",
            (email, password)
        )
        user = c.fetchone()

        if user:
            session["user_id"] = user[0]
            return redirect("/dashboard")
        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


# ── REGISTER ─────────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name     = request.form.get("name")
        email    = request.form.get("email")
        password = request.form.get("password")

        conn = get_db()
        c    = conn.cursor()
        try:
            c.execute(
                "INSERT INTO users (name, email, password) VALUES (?,?,?)",
                (name, email, password)
            )
            conn.commit()
        except Exception:
            return render_template("register.html", error="User already exists")

        return redirect("/login")

    return render_template("register.html")


# ── DASHBOARD ─────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db()
    c    = conn.cursor()
    c.execute("""
        SELECT probability, result, date
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
    """, (session["user_id"],))
    data = c.fetchall()
    return render_template("dashboard.html", data=data)


# ── PREDICTION — THE MAIN ROUTE ───────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
@login_required
def prediction():

    # GET request — just show the empty form
    if request.method != "POST":
        return render_template("predict.html")

    # ── Step 1: Collect form input ────────────────────────────
    patient_data = {}
    for feat in features:
        try:
            patient_data[feat] = float(request.form.get(feat, 0))
        except (ValueError, TypeError):
            patient_data[feat] = 0.0

    df     = pd.DataFrame([patient_data])
    scaled = scaler.transform(df)       # shape: (1, n_features)

    # ── Step 2: Raw model prediction ─────────────────────────
    prob = float(model.predict_proba(scaled)[0][1])
    prob = max(min(prob, 0.95), 0.01)

    # ── Step 3: Clinical adjustment (original logic) ──────────
    symptom_count = sum([
        patient_data.get("recent_accident",          0),
        patient_data.get("family_accident",          0),
        patient_data.get("witnessed_serious_injury", 0),
        patient_data.get("witnessed_corpse",         0),
        patient_data.get("traumatic_scene",          0),
        patient_data.get("psychiatric_history",      0),
    ])
    resilience = patient_data.get("psychological_resilience", 50)

    if symptom_count == 0:       prob = min(prob, 0.08)
    elif symptom_count == 1:     prob = min(prob, 0.35)
    elif symptom_count == 2:     prob = min(max(prob, 0.25), 0.60)
    elif symptom_count >= 3:     prob = max(prob, 0.50)

    if resilience >= 70:         prob *= 0.6
    elif resilience <= 30:       prob *= 1.2

    prob            = max(min(prob, 0.95), 0.01)
    probability_pct = round(prob * 100, 2)

    # ── Step 4: Risk label ────────────────────────────────────
    if probability_pct < 10:
        result, gauge_class = "Low PTSD Risk",      "low-risk"
    elif probability_pct < 30:
        result, gauge_class = "Moderate PTSD Risk", "moderate-risk"
    elif probability_pct < 60:
        result, gauge_class = "High PTSD Risk",     "high-risk"
    else:
        result, gauge_class = "Critical PTSD Risk", "critical-risk"

    # ── Step 5: Run all 3 XAI methods ─────────────────────────
    # This is what was MISSING in the original app.py
    print(f"\n--- Running XAI for {probability_pct}% PTSD prediction ---")

    shap_data, shap_plot, shap_method = run_shap(scaled)
    print(f"  ✅ SHAP done  ({shap_method})")

    lime_data, lime_plot, lime_method = run_lime(scaled)
    print(f"  ✅ LIME done  ({lime_method})")

    perm_data, perm_plot, perm_method = run_permutation(scaled)
    print(f"  ✅ Permutation done  ({perm_method})")

    # ── Step 6: Consensus — top features both SHAP & Perm agree on ──
    shap_top = [d["feature"] for d in shap_data[:3]]
    perm_top = [d["feature"] for d in perm_data[:3]]
    # Features that appear in top 3 of both methods
    agreed   = [f for f in shap_top if f in perm_top]
    # Fill up to 3 with SHAP top features if agreement is less
    consensus_top3 = list(dict.fromkeys(agreed + shap_top))[:3]

    # ── Step 7: Save to database ──────────────────────────────
    conn = get_db()
    c    = conn.cursor()
    c.execute("""
        INSERT INTO predictions (user_id, probability, result, date)
        VALUES (?, ?, ?, ?)
    """, (session["user_id"], probability_pct, result, str(datetime.now())))
    conn.commit()

    # ── Step 8: Pass EVERYTHING to predict.html ───────────────
    # This is the key fix — the original code never passed XAI variables
    return render_template(
        "predict.html",

        # ── Original prediction variables (unchanged) ─────────
        prediction_text = result,
        probability     = probability_pct,
        gauge_class     = gauge_class,

        # ── SHAP variables ────────────────────────────────────
        # 'explanations' keeps the existing {% if explanations %}
        # bar list in the result-card working
        explanations = shap_data,
        shap_data    = shap_data,
        shap_plot    = shap_plot,
        shap_method  = shap_method,

        # ── LIME variables ────────────────────────────────────
        lime_data   = lime_data,
        lime_plot   = lime_plot,
        lime_method = lime_method,

        # ── Permutation variables ─────────────────────────────
        perm_data   = perm_data,
        perm_plot   = perm_plot,
        perm_method = perm_method,

        # ── Consensus (features all methods agree on) ─────────
        consensus_top3 = consensus_top3,
    )


# ── LOGOUT ────────────────────────────────────────────────────
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ════════════════════════════════════════════════════════════
# RUN
# threaded=False prevents matplotlib thread safety issues
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, threaded=False)