"""
Sales Regression & Markdown Sensitivity Analysis
=================================================
Dataset: Walmart Recruiting - Store Sales Forecasting (Kaggle)

Business question:
  1. What actually drives weekly store sales (markdowns, holidays, weather,
     economic conditions)?
  2. At what markdown spend level does the promotional lift start flattening
     out — i.e. where's the point of diminishing returns on discounting?

Author: Rinit Jain
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# -----------------------------------------------------------------------
# 1. LOAD & MERGE
# -----------------------------------------------------------------------
# Raw Kaggle files are not checked into this repo (train.csv is ~13MB).
# Download them from the dataset link in the README and place them in
# data/raw/ before running this script.
train = pd.read_csv("data/raw/train.csv", parse_dates=["Date"])
features = pd.read_csv("data/raw/features.csv", parse_dates=["Date"])
stores = pd.read_csv("data/raw/stores.csv")

# features.csv already has IsHoliday -> drop duplicate before merge
features = features.drop(columns=["IsHoliday"])

# Aggregate train to Store-Date level (sum sales across all departments)
# -> gives us one "total weekly sales" figure per store, which is what a
#    business owner actually cares about for a markdown decision.
store_sales = (
    train.groupby(["Store", "Date"], as_index=False)
    .agg(Weekly_Sales=("Weekly_Sales", "sum"), IsHoliday=("IsHoliday", "max"))
)

df = store_sales.merge(features, on=["Store", "Date"], how="left")
df = df.merge(stores, on="Store", how="left")

# -----------------------------------------------------------------------
# 2. CLEAN & FEATURE ENGINEER
# -----------------------------------------------------------------------
markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
df[markdown_cols] = df[markdown_cols].fillna(0)
df["Total_Markdown"] = df[markdown_cols].sum(axis=1)

df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

df = df.dropna(subset=["CPI", "Unemployment"])  # a few stores missing econ data early on

print(f"Final modeling dataset: {df.shape[0]} store-weeks across {df['Store'].nunique()} stores")
df.to_csv("data/merged_store_weekly.csv", index=False)

# -----------------------------------------------------------------------
# 3. EDA CHARTS
# -----------------------------------------------------------------------
# 3a. Sales trend over time (all stores combined)
trend = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
plt.figure(figsize=(10, 4.5))
plt.plot(trend["Date"], trend["Weekly_Sales"], color="#2c7fb8")
plt.title("Total Weekly Sales Across All Stores (2010-2012)")
plt.ylabel("Weekly Sales ($)")
plt.xlabel("")
plt.tight_layout()
plt.savefig("charts/01_sales_trend.png")
plt.close()

# 3b. Sales by store type
plt.figure(figsize=(6, 4.5))
sns.boxplot(data=df, x="Type", y="Weekly_Sales", palette="Blues")
plt.title("Weekly Sales Distribution by Store Type")
plt.tight_layout()
plt.savefig("charts/02_sales_by_store_type.png")
plt.close()

# 3c. Correlation heatmap
corr_cols = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment",
             "Total_Markdown", "Size"]
plt.figure(figsize=(7, 5.5))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Key Drivers")
plt.tight_layout()
plt.savefig("charts/03_correlation_heatmap.png")
plt.close()

# -----------------------------------------------------------------------
# 4. REGRESSION MODEL
# -----------------------------------------------------------------------
model_df = df.copy()
model_df["IsHoliday"] = model_df["IsHoliday"].astype(int)
model_df = pd.get_dummies(model_df, columns=["Type"], drop_first=True)

feature_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Size",
                 "Total_Markdown", "IsHoliday", "Month"] + \
               [c for c in model_df.columns if c.startswith("Type_")]

X = model_df[feature_cols]
y = model_df["Weekly_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n--- Linear Regression ---")
print(f"R2:  {r2_score(y_test, lin_pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, lin_pred):,.0f}")

print("\n--- Random Forest (benchmark) ---")
print(f"R2:  {r2_score(y_test, rf_pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, rf_pred):,.0f}")

coef_table = pd.Series(lin_model.coef_, index=feature_cols).sort_values()
print("\nLinear Regression coefficients ($ impact per unit increase):")
print(coef_table)

plt.figure(figsize=(7, 5))
coef_table.plot(kind="barh", color=["#d73027" if v < 0 else "#1a9850" for v in coef_table])
plt.title("Regression Coefficients: What Moves Weekly Sales")
plt.xlabel("$ change in weekly sales per unit increase in feature")
plt.tight_layout()
plt.savefig("charts/04_regression_coefficients.png")
plt.close()

# Actual vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, rf_pred, alpha=0.3, s=10, color="#2c7fb8")
lims = [0, max(y_test.max(), rf_pred.max())]
plt.plot(lims, lims, "r--", linewidth=1)
plt.xlabel("Actual Weekly Sales")
plt.ylabel("Predicted Weekly Sales")
plt.title(f"Random Forest: Actual vs Predicted (R2 = {r2_score(y_test, rf_pred):.2f})")
plt.tight_layout()
plt.savefig("charts/05_actual_vs_predicted.png")
plt.close()

# -----------------------------------------------------------------------
# 5. SENSITIVITY ANALYSIS - "What if we spend more/less on markdowns?"
# -----------------------------------------------------------------------
# Rather than scoring a single synthetic "typical store" (which produces a
# noisy, step-shaped curve with a tree model), we take a large sample of
# REAL store-weeks, override just their Total_Markdown to each scenario
# value, and average the predictions. This holds every other real-world
# feature combination constant while isolating the effect of markdown
# spend, and averaging across many rows smooths out individual tree splits.
sample = X.sample(n=min(2000, len(X)), random_state=42).copy()

markdown_range = np.linspace(0, df["Total_Markdown"].quantile(0.95), 25)
scenario_results = []
for m in markdown_range:
    scenario = sample.copy()
    scenario["Total_Markdown"] = m
    avg_pred = rf_model.predict(scenario).mean()
    scenario_results.append({"Total_Markdown": m, "Predicted_Sales": avg_pred})

scenario_df = pd.DataFrame(scenario_results)
# Marginal lift per $1 of extra markdown (discrete derivative)
scenario_df["Marginal_Lift_per_Dollar"] = (
    scenario_df["Predicted_Sales"].diff() / scenario_df["Total_Markdown"].diff()
)

scenario_df.to_csv("data/markdown_sensitivity_table.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(scenario_df["Total_Markdown"], scenario_df["Predicted_Sales"], marker="o", color="#2c7fb8")
plt.xlabel("Total Markdown Spend ($)")
plt.ylabel("Predicted Weekly Sales ($)")
plt.title("Sensitivity: Predicted Sales vs Markdown Spend (typical store-week)")
plt.tight_layout()
plt.savefig("charts/06_markdown_sensitivity_curve.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(scenario_df["Total_Markdown"][1:], scenario_df["Marginal_Lift_per_Dollar"][1:],
          marker="o", color="#d95f0e")
plt.axhline(0, color="grey", linewidth=0.8)
plt.xlabel("Total Markdown Spend ($)")
plt.ylabel("Marginal $ Sales Lift per $1 of Markdown")
plt.title("Diminishing Returns: Marginal Return on Markdown Spend")
plt.tight_layout()
plt.savefig("charts/07_marginal_return_curve.png")
plt.close()

# Identify the point where marginal lift drops below $1 (i.e. spend stops paying for itself)
below_1 = scenario_df[scenario_df["Marginal_Lift_per_Dollar"] < 1]
avg_marginal = scenario_df["Marginal_Lift_per_Dollar"].mean()
if len(below_1) == len(scenario_df) - 1:  # true for essentially the whole range
    print(f"\nMarginal lift stays below $1 of sales per $1 of markdown across the "
          f"entire range tested (avg ${avg_marginal:.2f} per $1). Markdown spend "
          f"alone does not appear to be revenue-additive at the store level in "
          f"this dataset -- it likely serves other goals (inventory clearance, "
          f"competitive matching) rather than driving incremental sales.")
else:
    tipping_point = below_1["Total_Markdown"].iloc[0]
    print(f"\nEstimated tipping point (marginal lift < $1 per $1 spent): ${tipping_point:,.0f}")

print("\nDone. Charts saved to /charts, tables saved as CSV.")
