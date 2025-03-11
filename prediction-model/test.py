# test.py
import pandas as pd
import joblib
from datetime import datetime

# Load the full-data models
clf_pipeline = joblib.load("train2_classifier.pkl")
reg_pipeline_home = joblib.load("train2_regressor_home.pkl")
reg_pipeline_away = joblib.load("train2_regressor_away.pkl")

# Create a test instance for the match "Barcelona vs Real Madrid"
# We need to create a dictionary with all features used in training:
# For demonstration, we'll assume:
# - Date: today's date (or a fixed date, e.g., "2025-01-01")
# - full_time: "F"
# - competition: For example, "la_liga"
# - home: "Barcelona"
# - away: "Real Madrid"
# - home_country and away_country: "spain"
# - level: "club" (since these are club teams)

test_data = {
    "date": ["2025-03-09T00:00:00.000Z"],
    "full_time": ["F"],
    "competition": ["la_liga"],
    "home": ["Getafe"],
    "away": ["Atletico Madrid"],
    "home_country": ["spain"],
    "away_country": ["spain"],
    "level": ["club"]
}

# Convert to DataFrame
test_df = pd.DataFrame(test_data)

# Process date: extract year, month, and day_of_week
test_df["date"] = pd.to_datetime(test_df["date"])
test_df["year"] = test_df["date"].dt.year
test_df["month"] = test_df["date"].dt.month
test_df["day_of_week"] = test_df["date"].dt.dayofweek

# Select the same features as in training
features = [
    "year", "month", "day_of_week",
    "full_time", "competition",
    "home", "away",
    "home_country", "away_country",
    "level"
]
X_test = test_df[features]

# --- Make Predictions ---
# Predict outcome classification
result_pred = clf_pipeline.predict(X_test)[0]
# Get prediction probabilities to see confidence level
result_probs = clf_pipeline.predict_proba(X_test)[0]

# Get raw score predictions before rounding
raw_home_goals = reg_pipeline_home.predict(X_test)[0]
raw_away_goals = reg_pipeline_away.predict(X_test)[0]

# Round the predicted goals to the nearest integer
pred_home_goals = round(raw_home_goals)
pred_away_goals = round(raw_away_goals)

# Determine outcome based on the predicted scores for consistency check
score_based_outcome = "Home Win" if pred_home_goals > pred_away_goals else \
                     ("Away Win" if pred_home_goals < pred_away_goals else "Draw")

# Map numeric result to human-readable string using the new mapping:
# 0 = Away Win, 1 = Draw, 2 = Home Win
if result_pred == 2:
    model_outcome = "Home Win"
elif result_pred == 1:
    model_outcome = "Draw"
elif result_pred == 0:
    model_outcome = "Away Win"

print("Model Predicted Outcome:", model_outcome)
print(f"Outcome Probabilities (Away Win/Draw/Home Win): {result_probs}")
print(f"Raw Goal Predictions - Home: {raw_home_goals:.2f}, Away: {raw_away_goals:.2f}")
print(f"Rounded Score: {pred_home_goals} - {pred_away_goals}")
print(f"Score-based Outcome: {score_based_outcome}")

# Check if there's a mismatch
if model_outcome != score_based_outcome:
    print("\n⚠️ WARNING: Inconsistency detected between outcome and score predictions!")
    print(f"  • Classification model predicts: {model_outcome}")
    print(f"  • Score predictions suggest: {score_based_outcome}")
    print("\nConsider using the score-based outcome for more consistency.")
