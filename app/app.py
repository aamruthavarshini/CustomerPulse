from flask import Flask, render_template, request, session, redirect
import joblib
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np

app = Flask(__name__)
app.secret_key = "customerpulse_secret"

# ==========================
# PATH CONFIGURATION
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "../src/churn_pipeline.pkl")
data_path = os.path.join(BASE_DIR, "../data/raw/telco_churn.csv")

pipeline = joblib.load(model_path)


# ==========================
# LOGIN PAGE
# ==========================
@app.route("/")
def login():
    return render_template("login.html")

@app.route("/login_role", methods=["POST"])
def login_role():
    role = request.form.get("role")
    if role not in ["manager", "analyst"]:
        return redirect("/")
    session["role"] = role
    return redirect("/dashboard")


# ==========================
# DASHBOARD PAGE
# ==========================
@app.route("/dashboard")
def dashboard():
    if "role" not in session:
        return redirect("/")

    role = session.get("role", "analyst")

    df = pd.read_csv(data_path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    X = df.drop("Churn", axis=1)
    probs = pipeline.predict_proba(X)[:, 1]
    df["ChurnProbability"] = probs

    df["Risk"] = df["ChurnProbability"].apply(
        lambda p: "High" if p > 0.7 else "Medium" if p > 0.4 else "Low"
    )

    df["CustomerValue"] = df["MonthlyCharges"] * df["tenure"]
    df["PriorityScore"] = df["ChurnProbability"] * df["CustomerValue"]

    # Recommended Actions
    def recommend_action(row):
        if row["Contract"] == "Month-to-month":
            return "Offer discount for long-term contract"
        elif row["tenure"] < 6:
            return "Early engagement follow-up call"
        elif row["MonthlyCharges"] > df["MonthlyCharges"].median():
            return "Offer loyalty reward"
        else:
            return "Send retention email campaign"

    df["RecommendedAction"] = df.apply(recommend_action, axis=1)

    total = len(df)
    high = len(df[df["Risk"] == "High"])
    medium = len(df[df["Risk"] == "Medium"])
    low = len(df[df["Risk"] == "Low"])

    top10 = df.sort_values("PriorityScore", ascending=False).head(10)

    contract_risk = (
        df.groupby("Contract")["ChurnProbability"]
        .mean()
        .round(3)
        .to_dict()
    )
    highest_contract = max(contract_risk, key=contract_risk.get)

    revenue_at_risk = df[df["Risk"] == "High"]["MonthlyCharges"].sum()
    retention_savings = revenue_at_risk * 0.30

    retention_cost = 500
    high_risk_customers = df[df["Risk"] == "High"]
    estimated_saved = int(len(high_risk_customers) * 0.30)
    total_retention_cost = estimated_saved * retention_cost
    roi = retention_savings - total_retention_cost

    # Currency formatting (IMPORTANT FIX)
    revenue_at_risk_fmt = "{:,.2f}".format(revenue_at_risk)
    retention_savings_fmt = "{:,.2f}".format(retention_savings)
    total_retention_cost_fmt = "{:,.2f}".format(total_retention_cost)
    roi_fmt = "{:,.2f}".format(roi)

    return render_template(
        "dashboard.html",
        role=role,
        total=total,
        high=high,
        medium=medium,
        low=low,
        top10=top10.to_dict(orient="records"),
        contract_risk=contract_risk,
        revenue_at_risk=revenue_at_risk_fmt,
        retention_savings=retention_savings_fmt,
        total_retention_cost=total_retention_cost_fmt,
        roi=roi_fmt,
        highest_contract=highest_contract,
    )



# ==========================
# STRATEGY SIMULATOR
# ==========================
@app.route("/strategy", methods=["GET", "POST"])
def strategy():
    if "role" not in session:
        return redirect("/")

    if session.get("role") != "analyst":
        return redirect("/dashboard")

    df = pd.read_csv(data_path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    X = df.drop("Churn", axis=1)
    probs = pipeline.predict_proba(X)[:, 1]
    df["ChurnProbability"] = probs

    df["Risk"] = df["ChurnProbability"].apply(
        lambda p: "High" if p > 0.7 else "Medium" if p > 0.4 else "Low"
    )

    high_risk = df[df["Risk"] == "High"]
    revenue_at_risk = high_risk["MonthlyCharges"].sum()

    recovery_rate = 30
    retention_cost = 500

    if request.method == "POST":
        recovery_rate = int(request.form["recovery_rate"])
        retention_cost = int(request.form["retention_cost"])

    estimated_saved = int(len(high_risk) * (recovery_rate / 100))
    expected_revenue_saved = revenue_at_risk * (recovery_rate / 100)
    total_campaign_cost = estimated_saved * retention_cost
    roi = expected_revenue_saved - total_campaign_cost
    roi_raw = roi

    # Currency formatting fix
    revenue_at_risk_fmt = "{:,.2f}".format(revenue_at_risk)
    expected_revenue_saved_fmt = "{:,.2f}".format(expected_revenue_saved)
    total_campaign_cost_fmt = "{:,.2f}".format(total_campaign_cost)
    roi_fmt = "{:,.2f}".format(roi)

    return render_template(
    "strategy.html",
    recovery_rate=recovery_rate,
    retention_cost=retention_cost,
    revenue_at_risk=revenue_at_risk_fmt,
    expected_revenue_saved=expected_revenue_saved_fmt,
    total_campaign_cost=total_campaign_cost_fmt,
    roi=roi_fmt,
    roi_raw=roi_raw,
)


# ==========================
# ANALYTICS PAGE (FIXED)
# ==========================
@app.route("/analytics")
def analytics():
    if "role" not in session:
        return redirect("/")

    if session.get("role") != "analyst":
        return redirect("/dashboard")

    df = pd.read_csv(data_path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    y_pred = pipeline.predict(X)

    accuracy = round(accuracy_score(y, y_pred), 3)
    precision = round(precision_score(y, y_pred), 3)
    recall = round(recall_score(y, y_pred), 3)

    cm = confusion_matrix(y, y_pred)

    # Feature importance (Logistic Regression only)
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importance = model.coef_[0]

    # Sort by absolute importance
    sorted_features = sorted(
        zip(feature_names, importance),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]

    cleaned_features = []

    for name, coef in sorted_features:
        clean_name = name.replace("num__", "")
        clean_name = clean_name.replace("cat__", "")
        clean_name = clean_name.replace("_", " ")
        clean_name = clean_name.title()

        cleaned_features.append((clean_name, round(coef, 3)))

    return render_template(
        "analytics.html",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        cm=cm,
        feature_importance=cleaned_features
    )
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)