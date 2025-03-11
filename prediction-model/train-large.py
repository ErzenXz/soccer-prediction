# train2.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from lightgbm import LGBMClassifier, LGBMRegressor
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import time
from tqdm import tqdm
from datetime import timedelta

# Function to estimate and display ETA
class ETATracker:
    def __init__(self, total_steps, name="Training"):
        self.total_steps = total_steps
        self.start_time = time.time()
        self.step_times = []
        self.name = name
        print(f"\n{self.name} Progress:")
    
    def update(self, step, step_name=""):
        elapsed = time.time() - self.start_time
        self.step_times.append(elapsed)
        
        # Calculate average time per step and estimate remaining time
        avg_step_time = elapsed / step
        eta_seconds = avg_step_time * (self.total_steps - step)
        eta = str(timedelta(seconds=int(eta_seconds)))
        
        # Display progress
        percent = int(100 * step / self.total_steps)
        progress_bar = "=" * (percent // 5) + ">" + " " * (20 - percent // 5)
        print(f"\r[{progress_bar}] {percent}% - Step {step}/{self.total_steps}: {step_name} - ETA: {eta}", end="")
        
        if step == self.total_steps:
            print("\nTraining complete!")
            total_time = str(timedelta(seconds=int(elapsed)))
            print(f"Total training time: {total_time}")

# Define steps for train2.py (training on full dataset)
total_steps = 10
eta_tracker = ETATracker(total_steps, "Train2")

print("Loading data and preparing features...")
# Load the data from the parquet file
data = pd.read_parquet("./data/games.parquet")
eta_tracker.update(1, "Data loading")

# Create the target variable for classification:
data["result"] = data.apply(lambda row: 1 if row["gh"] > row["ga"]
                              else (-1 if row["gh"] < row["ga"] else 0), axis=1)
data["home_goals"] = data["gh"]
data["away_goals"] = data["ga"]

# --- Feature Engineering ---
print("Feature engineering...")
data["date"] = pd.to_datetime(data["date"])
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day_of_week"] = data["date"].dt.dayofweek
data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

# Sort data chronologically (needed for form features)
data = data.sort_values("date")
eta_tracker.update(2, "Basic feature engineering")

# --- Team Form Features ---
print("Calculating team form...")
def calculate_team_form(df, team_col, result_col, window=5):
    teams = df[team_col].unique()
    form_dict = {}
    
    for team in tqdm(teams, desc=f"{team_col} form", leave=False):
        team_matches = df[df[team_col] == team].copy()
        if team_col == 'home':
            team_matches['result_adj'] = team_matches[result_col]
        else:
            team_matches['result_adj'] = -team_matches[result_col]
        
        team_matches['rolling_form'] = team_matches['result_adj'].rolling(window, min_periods=1).mean()
        form_dict[team] = dict(zip(team_matches['date'], team_matches['rolling_form']))
    
    return form_dict

# Calculate team forms
home_forms = calculate_team_form(data, 'home', 'result')
away_forms = calculate_team_form(data, 'away', 'result')

# Add form features to dataset
data['home_form'] = data.apply(lambda x: home_forms.get(x['home'], {}).get(x['date'], 0), axis=1)
data['away_form'] = data.apply(lambda x: away_forms.get(x['away'], {}).get(x['date'], 0), axis=1)
eta_tracker.update(3, "Team form calculation")

# Set feature list
features = [
    "year", "month", "day_of_week", "is_weekend", 
    "full_time", "competition",
    "home", "away",
    "home_country", "away_country",
    "level",
    "home_form", "away_form"
]

X = data[features]
y_class = data["result"].map({-1: 0, 0: 1, 1: 2})  # Transform to 0,1,2
y_home = data["home_goals"]
y_away = data["away_goals"]
eta_tracker.update(4, "Feature preparation")

# --- Preprocessing Pipeline ---
categorical_features = ["full_time", "competition", "home", "away", "home_country", "away_country", "level"]
numerical_features = [col for col in features if col not in categorical_features]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numerical_features)
])
eta_tracker.update(5, "Preprocessing setup")

# --- Build and Train XGBoost Classifier ---
print("Training XGBoost classification model...")
xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb.XGBClassifier(objective="multi:softmax", num_class=3, random_state=42))
])
xgb_pipeline.fit(X, y_class)
eta_tracker.update(6, "XGBoost classification model")

# --- Build and Train LightGBM Classifier ---
print("Training LightGBM classification model...")
lgbm_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LGBMClassifier(objective="multiclass", num_class=3, random_state=42))
])
lgbm_pipeline.fit(X, y_class)
eta_tracker.update(7, "LightGBM classification model")

# --- Build and Train Regression Pipelines ---
print("Training XGBoost regression models...")
# For home goals
xgb_reg_home = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", xgb.XGBRegressor(objective="reg:squarederror", random_state=42))
])
xgb_reg_home.fit(X, y_home)

# For away goals
xgb_reg_away = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", xgb.XGBRegressor(objective="reg:squarederror", random_state=42))
])
xgb_reg_away.fit(X, y_away)
eta_tracker.update(8, "XGBoost regression models")

print("Training LightGBM regression models...")
# LightGBM models
lgbm_reg_home = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LGBMRegressor(objective="regression", random_state=42))
])
lgbm_reg_home.fit(X, y_home)

lgbm_reg_away = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LGBMRegressor(objective="regression", random_state=42))
])
lgbm_reg_away.fit(X, y_away)
eta_tracker.update(9, "LightGBM regression models")

# Save all models
print("Saving trained models...")
# Save the XGBoost models for backward compatibility
joblib.dump(xgb_pipeline, "train2_classifier.pkl")
joblib.dump(xgb_reg_home, "train2_regressor_home.pkl")
joblib.dump(xgb_reg_away, "train2_regressor_away.pkl")

# Save individual and ensemble models
joblib.dump(xgb_pipeline, "train2_classifier_xgb.pkl")
joblib.dump(lgbm_pipeline, "train2_classifier_lgbm.pkl")
joblib.dump((xgb_reg_home, lgbm_reg_home), "train2_regressor_home_ensemble.pkl")
joblib.dump((xgb_reg_away, lgbm_reg_away), "train2_regressor_away_ensemble.pkl")
eta_tracker.update(10, "Saving models")

# --- Evaluate models ---
print("\n=== Model Evaluation ===")

# Make predictions on the training data (in-sample performance)
y_class_pred_xgb = xgb_pipeline.predict(X)
y_class_pred_lgbm = lgbm_pipeline.predict(X)
y_class_proba_xgb = xgb_pipeline.predict_proba(X)
y_class_proba_lgbm = lgbm_pipeline.predict_proba(X)

# Ensemble predictions using average probability
ensemble_proba = (y_class_proba_xgb + y_class_proba_lgbm) / 2
ensemble_pred = np.argmax(ensemble_proba, axis=1)

# Calculate classification metrics
xgb_accuracy = accuracy_score(y_class, y_class_pred_xgb)
lgbm_accuracy = accuracy_score(y_class, y_class_pred_lgbm)
ensemble_accuracy = accuracy_score(y_class, ensemble_pred)

print(f"Classification Accuracy - XGBoost: {xgb_accuracy:.4f}")
print(f"Classification Accuracy - LGBM: {lgbm_accuracy:.4f}")
print(f"Classification Accuracy - Ensemble: {ensemble_accuracy:.4f}")

# Regression predictions
y_home_pred_xgb = xgb_reg_home.predict(X)
y_home_pred_lgbm = lgbm_reg_home.predict(X)
y_away_pred_xgb = xgb_reg_away.predict(X)
y_away_pred_lgbm = lgbm_reg_away.predict(X)

# Ensemble regression predictions
y_home_pred_ensemble = (y_home_pred_xgb + y_home_pred_lgbm) / 2
y_away_pred_ensemble = (y_away_pred_xgb + y_away_pred_lgbm) / 2

# Calculate regression metrics
home_rmse_xgb = np.sqrt(mean_squared_error(y_home, y_home_pred_xgb))
home_rmse_lgbm = np.sqrt(mean_squared_error(y_home, y_home_pred_lgbm))
home_rmse_ensemble = np.sqrt(mean_squared_error(y_home, y_home_pred_ensemble))

away_rmse_xgb = np.sqrt(mean_squared_error(y_away, y_away_pred_xgb))
away_rmse_lgbm = np.sqrt(mean_squared_error(y_away, y_away_pred_lgbm))
away_rmse_ensemble = np.sqrt(mean_squared_error(y_away, y_away_pred_ensemble))

print(f"\nHome Goals RMSE - XGBoost: {home_rmse_xgb:.4f}")
print(f"Home Goals RMSE - LGBM: {home_rmse_lgbm:.4f}")
print(f"Home Goals RMSE - Ensemble: {home_rmse_ensemble:.4f}")

print(f"\nAway Goals RMSE - XGBoost: {away_rmse_xgb:.4f}")
print(f"Away Goals RMSE - LGBM: {away_rmse_lgbm:.4f}")
print(f"Away Goals RMSE - Ensemble: {away_rmse_ensemble:.4f}")

# Check consistency between score and outcome predictions
rounded_home_goals = np.round(y_home_pred_ensemble)
rounded_away_goals = np.round(y_away_pred_ensemble)

# Generate outcomes from scores (0=away win, 1=draw, 2=home win)
score_outcomes = np.where(rounded_home_goals > rounded_away_goals, 2,
                         np.where(rounded_home_goals < rounded_away_goals, 0, 1))

consistency = accuracy_score(score_outcomes, ensemble_pred)
print(f"\nScore-Outcome Consistency: {consistency:.4f}")

# Show detailed sample predictions
print("\n=== Sample Predictions ===")
sample_indices = np.random.choice(len(X), 10, replace=False)
sample_X = X.iloc[sample_indices]
sample_data = data.iloc[sample_indices]

# Create DataFrame with predictions
sample_teams = pd.DataFrame({
    'Home': sample_X['home'],
    'Away': sample_X['away'],
    'Competition': sample_X['competition'],
    'Date': sample_data['date'].dt.strftime('%Y-%m-%d'),
    'Actual Result': [
        "Home Win" if r == 2 else "Draw" if r == 1 else "Away Win" 
        for r in y_class.iloc[sample_indices]
    ],
    'Predicted Result': [
        "Home Win" if r == 2 else "Draw" if r == 1 else "Away Win" 
        for r in ensemble_pred[sample_indices]
    ],
    'Result Probability': [f"{max(p):.2f}" for p in ensemble_proba[sample_indices]],
    'Actual Score': [
        f"{h} - {a}" for h, a in zip(
            y_home.iloc[sample_indices], 
            y_away.iloc[sample_indices]
        )
    ],
    'Predicted Score': [
        f"{round(h)} - {round(a)}" for h, a in zip(
            y_home_pred_ensemble[sample_indices], 
            y_away_pred_ensemble[sample_indices]
        )
    ]
})

# Display sample predictions with nice formatting
print(sample_teams.to_string(index=False))

# Optional: Save evaluation results
eval_results = {
    'xgb_accuracy': xgb_accuracy,
    'lgbm_accuracy': lgbm_accuracy,
    'ensemble_accuracy': ensemble_accuracy,
    'home_rmse_ensemble': home_rmse_ensemble,
    'away_rmse_ensemble': away_rmse_ensemble,
    'score_outcome_consistency': consistency
}

# Save evaluation results for future reference
joblib.dump(eval_results, "train2_evaluation.pkl")

print("\nTrain2 complete: Enhanced models trained on the full dataset and saved.")
