import os
import numpy as np
import pandas as pd

# Eğer CSV dosyası yoksa otomatik oluştur
if not os.path.exists("student_performance.csv"):
    np.random.seed(42)
    df = pd.DataFrame({
        "gender": np.random.choice(["Male", "Female"], 1000),
        "study_hours": np.random.randint(1, 30, 1000),
        "attendance": np.random.randint(60, 100, 1000),
        "previous_score": np.random.randint(40, 90, 1000),
        "assignments_completed": np.random.randint(1, 10, 1000),
        "final_score": np.random.randint(50, 100, 1000),
    })
    df.to_csv("student_performance.csv", index=False)
"""
Student Performance Data Analysis
Author: Your Name
Description:
Analyzes student performance data, performs feature engineering,
and builds regression models to predict final scores.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
# Expected CSV columns:
# gender, study_hours, attendance, previous_score,
# assignments_completed, final_score

df = pd.read_csv("student_performance.csv")

print("Dataset shape:", df.shape)
print(df.head())


# --------------------------------------------------
# 2. DATA CLEANING
# --------------------------------------------------
# Handle missing values
df.dropna(inplace=True)

# Encode categorical variables
df["gender"] = df["gender"].map({"Male": 0, "Female": 1})


# --------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------
# Study efficiency: how effectively time is used
df["study_efficiency"] = df["study_hours"] / (df["attendance"] + 1)

# Engagement score: weighted academic engagement
df["engagement_score"] = (
    0.5 * df["attendance"] +
    0.5 * df["assignments_completed"]
)

print("\nNew features added:")
print(df[["study_efficiency", "engagement_score"]].head())


# --------------------------------------------------
# 4. FEATURE / TARGET SPLIT
# --------------------------------------------------
X = df.drop("final_score", axis=1)
y = df["final_score"]


# --------------------------------------------------
# 5. TRAIN / TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --------------------------------------------------
# 6. BASELINE MODEL (WITHOUT SCALING)
# --------------------------------------------------
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

baseline_preds = baseline_model.predict(X_test)
baseline_r2 = r2_score(y_test, baseline_preds)

print("\nBaseline Linear Regression R²:", round(baseline_r2, 3))


# --------------------------------------------------
# 7. FEATURE SCALING
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 8. LINEAR REGRESSION (IMPROVED)
# --------------------------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

lr_preds = lr_model.predict(X_test_scaled)
lr_r2 = r2_score(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

print("\nImproved Linear Regression Performance")
print("R² Score:", round(lr_r2, 3))
print("RMSE:", round(lr_rmse, 2))


# --------------------------------------------------
# 9. RANDOM FOREST REGRESSION
# --------------------------------------------------
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

print("\nRandom Forest Performance")
print("R² Score:", round(rf_r2, 3))
print("RMSE:", round(rf_rmse, 2))


# --------------------------------------------------
# 10. PERFORMANCE IMPROVEMENT
# --------------------------------------------------
improvement = ((lr_r2 - baseline_r2) / baseline_r2) * 100
print(f"\nModel performance improved by approximately {improvement:.1f}% (R²)")


# --------------------------------------------------
# 11. FEATURE IMPORTANCE (RANDOM FOREST)
# --------------------------------------------------
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance)


# --------------------------------------------------
# 12. VISUALIZATION
# --------------------------------------------------
plt.figure(figsize=(8, 5))
feature_importance.plot(kind="bar")
plt.title("Feature Importance - Random Forest")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
