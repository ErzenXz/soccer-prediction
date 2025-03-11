import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from lightgbm import LGBMClassifier, LGBMRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import time
from tqdm import tqdm

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

# Define total steps for train1.py
total_steps = 12  # Total major steps in the training process
eta_tracker = ETATracker(total_steps, "Train1")

print("Loading and preparing data...")
# Load the data from the parquet file
start_time = time.time()
data = pd.read_parquet("./data/games.parquet")
eta_tracker.update(1, "Data loading")

# Create the target variable for classification:
data["result"] = data.apply(lambda row: 1 if row["gh"] > row["ga"]
                              else (-1 if row["gh"] < row["ga"] else 0), axis=1)
data["home_goals"] = data["gh"]
data["away_goals"] = data["ga"]

# --- Advanced Feature Engineering ---
print("Performing advanced feature engineering...")

# Convert the date column to datetime and extract temporal features
data["date"] = pd.to_datetime(data["date"])
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day_of_week"] = data["date"].dt.dayofweek
data["day_of_year"] = data["date"].dt.dayofyear
data["week_of_year"] = data["date"].dt.isocalendar().week
data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)  # Weekend flag (Sat/Sun)

# Sort data chronologically for time-based features
data = data.sort_values("date")
eta_tracker.update(2, "Basic feature engineering")

# --- Team Form Features ---
# Calculate rolling form for each team (last 5 matches)
def calculate_team_form(df, team_col, result_col, window=5):
    teams = df[team_col].unique()
    form_dict = {}
    
    for team in tqdm(teams, desc=f"{team_col} form", leave=False):
        team_matches = df[df[team_col] == team].copy()
        if team_col == 'home':
            # For home team: 1=win, 0=draw, -1=loss
            team_matches['result_adj'] = team_matches[result_col]
        else:
            # For away team: -1=win, 0=draw, 1=loss (perspective flip)
            team_matches['result_adj'] = -team_matches[result_col]
        
        # Calculate rolling mean of adjusted results
        team_matches['rolling_form'] = team_matches['result_adj'].rolling(window, min_periods=1).mean()
        
        # Store the form in dictionary
        form_dict[team] = dict(zip(team_matches['date'], team_matches['rolling_form']))
    
    return form_dict

# Calculate team forms
print("Calculating team form...")
home_forms = calculate_team_form(data, 'home', 'result')
away_forms = calculate_team_form(data, 'away', 'result')

# Add form features to dataset
data['home_form'] = data.apply(lambda x: home_forms.get(x['home'], {}).get(x['date'], 0), axis=1)
data['away_form'] = data.apply(lambda x: away_forms.get(x['away'], {}).get(x['date'], 0), axis=1)
eta_tracker.update(3, "Team form calculation")

# Add goals-based form
def calculate_goal_form(df, team_col, goals_col, window=5):
    teams = df[team_col].unique()
    form_dict = {}
    
    for team in tqdm(teams, desc=f"{team_col} {goals_col}", leave=False):
        team_matches = df[df[team_col] == team].copy()
        team_matches['rolling_goals'] = team_matches[goals_col].rolling(window, min_periods=1).mean()
        form_dict[team] = dict(zip(team_matches['date'], team_matches['rolling_goals']))
    
    return form_dict

print("Calculating goal statistics...")
home_goal_forms = calculate_goal_form(data, 'home', 'home_goals')
away_goal_forms = calculate_goal_form(data, 'away', 'away_goals')
home_conceded_forms = calculate_goal_form(data, 'home', 'away_goals')  # Goals conceded at home
away_conceded_forms = calculate_goal_form(data, 'away', 'home_goals')  # Goals conceded away

data['home_goals_form'] = data.apply(lambda x: home_goal_forms.get(x['home'], {}).get(x['date'], 0), axis=1)
data['away_goals_form'] = data.apply(lambda x: away_goal_forms.get(x['away'], {}).get(x['date'], 0), axis=1)
data['home_conceded_form'] = data.apply(lambda x: home_conceded_forms.get(x['home'], {}).get(x['date'], 0), axis=1)
data['away_conceded_form'] = data.apply(lambda x: away_conceded_forms.get(x['away'], {}).get(x['date'], 0), axis=1)
eta_tracker.update(4, "Goal statistics calculation")

# --- Head-to-Head Features ---
print("Calculating head-to-head statistics...")
def get_h2h_stats(df, home_team, away_team, match_date):
    # Get historical matches between these teams before current date
    h2h_matches = df[((df['home'] == home_team) & (df['away'] == away_team) | 
                    (df['home'] == away_team) & (df['away'] == home_team)) &
                    (df['date'] < match_date)]
    
    if len(h2h_matches) == 0:
        return 0, 0, 0  # No previous meetings
    
    # Count results from home_team perspective
    home_team_home_matches = h2h_matches[(h2h_matches['home'] == home_team)]
    home_team_away_matches = h2h_matches[(h2h_matches['away'] == home_team)]
    
    home_wins = sum(home_team_home_matches['result'] == 1)
    home_draws = sum(home_team_home_matches['result'] == 0)
    home_losses = sum(home_team_home_matches['result'] == -1)
    
    away_wins = sum(home_team_away_matches['result'] == -1)
    away_draws = sum(home_team_away_matches['result'] == 0)
    away_losses = sum(home_team_away_matches['result'] == 1)
    
    wins = home_wins + away_wins
    draws = home_draws + away_draws
    losses = home_losses + away_losses
    
    total = len(h2h_matches)
    if total == 0:
        win_rate = 0
    else:
        win_rate = (wins - losses) / total  # Positive means home team dominates, negative means away team
    
    # Recent form - last match result
    last_match = h2h_matches.loc[h2h_matches['date'].idxmax()]
    if last_match['home'] == home_team:
        last_result = last_match['result']  # 1, 0, -1
    else:
        last_result = -last_match['result']  # Flip perspective
    
    # Average goals
    home_goals = home_team_home_matches['gh'].mean() if len(home_team_home_matches) > 0 else 0
    home_conceded = home_team_home_matches['ga'].mean() if len(home_team_home_matches) > 0 else 0
    away_goals = home_team_away_matches['ga'].mean() if len(home_team_away_matches) > 0 else 0
    away_conceded = home_team_away_matches['gh'].mean() if len(home_team_away_matches) > 0 else 0
    
    goal_advantage = (home_goals + away_goals) - (home_conceded + away_conceded)
    
    return win_rate, last_result, goal_advantage

# Apply H2H calculations with progress bar
h2h_results = []
for i, row in tqdm(list(data.iterrows()), desc="H2H calculations"):
    win_rate, last_result, goal_adv = get_h2h_stats(data, row['home'], row['away'], row['date'])
    h2h_results.append((win_rate, last_result, goal_adv))

data['h2h_win_rate'] = [result[0] for result in h2h_results]
data['h2h_last_result'] = [result[1] for result in h2h_results]
data['h2h_goal_advantage'] = [result[2] for result in h2h_results]
eta_tracker.update(5, "Head-to-head calculations")

# Final feature set
features = [
    "year", "month", "day_of_week", "day_of_year", "week_of_year", "is_weekend",
    "full_time", "competition",
    "home", "away",
    "home_country", "away_country",
    "level",
    "home_form", "away_form",
    "home_goals_form", "away_goals_form",
    "home_conceded_form", "away_conceded_form",
    "h2h_win_rate", "h2h_last_result", "h2h_goal_advantage"
]

# Prepare feature matrix and targets
print("Preparing feature matrix and targets...")
X = data[features]
y_class = data["result"].map({-1: 0, 0: 1, 1: 2})  # Transform to 0,1,2 for XGBoost
y_home = data["home_goals"]
y_away = data["away_goals"]

# Split the data: 70% training, 30% held out.
X_train, X_valid, y_class_train, y_class_valid, y_home_train, y_home_valid, y_away_train, y_away_valid = train_test_split(
    X, y_class, y_home, y_away, test_size=0.3, random_state=42, stratify=y_class)
eta_tracker.update(6, "Data split and preparation")

# --- Preprocessing Pipeline ---
print("Setting up preprocessing pipeline...")
categorical_features = ["full_time", "competition", "home", "away", "home_country", "away_country", "level"]
numerical_features = [col for col in features if col not in categorical_features]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numerical_features)
])

# --- Cross-validation strategy ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Hyperparameter Tuning for Classification ---
print("Training and tuning classification model...")
xgb_params = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__max_depth': [3, 5, 7],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0],
}

clf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb.XGBClassifier(objective="multi:softmax", num_class=3, random_state=42))
])

print("Starting grid search (this may take a while)...")
grid_search = GridSearchCV(
    clf_pipeline, 
    xgb_params, 
    cv=cv, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_class_train)

best_clf_pipeline = grid_search.best_estimator_
print(f"Best classification parameters: {grid_search.best_params_}")
eta_tracker.update(7, "Classification model tuning")

# --- Train ensemble of models for home goals ---
print("Training home goals regression model...")
reg_pipeline_home = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", xgb.XGBRegressor(objective="reg:squarederror", random_state=42))
])

# Second model in ensemble (LightGBM)
reg_pipeline_home_lgbm = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LGBMRegressor(objective="regression", random_state=42))
])

reg_pipeline_home.fit(X_train, y_home_train)
eta_tracker.update(8, "Home goals XGBoost model")

reg_pipeline_home_lgbm.fit(X_train, y_home_train)
eta_tracker.update(9, "Home goals LightGBM model")

# --- Train ensemble of models for away goals ---
print("Training away goals regression model...")
reg_pipeline_away = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", xgb.XGBRegressor(objective="reg:squarederror", random_state=42))
])

# Second model in ensemble (LightGBM)
reg_pipeline_away_lgbm = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LGBMRegressor(objective="regression", random_state=42))
])

reg_pipeline_away.fit(X_train, y_away_train)
eta_tracker.update(10, "Away goals XGBoost model")

reg_pipeline_away_lgbm.fit(X_train, y_away_train)
eta_tracker.update(11, "Away goals LightGBM model")

# --- Evaluate on validation set ---
print("Evaluating models on validation set...")
# Classification metrics
y_class_pred = best_clf_pipeline.predict(X_valid)
classification_accuracy = accuracy_score(y_class_valid, y_class_pred)
f1 = f1_score(y_class_valid, y_class_pred, average='weighted')

# Regression metrics - XGBoost
y_home_pred_xgb = reg_pipeline_home.predict(X_valid)
y_away_pred_xgb = reg_pipeline_away.predict(X_valid)
home_mae_xgb = mean_absolute_error(y_home_valid, y_home_pred_xgb)
home_rmse_xgb = np.sqrt(mean_squared_error(y_home_valid, y_home_pred_xgb))
away_mae_xgb = mean_absolute_error(y_away_valid, y_away_pred_xgb)
away_rmse_xgb = np.sqrt(mean_squared_error(y_away_valid, y_away_pred_xgb))

# Regression metrics - LGBM
y_home_pred_lgbm = reg_pipeline_home_lgbm.predict(X_valid)
y_away_pred_lgbm = reg_pipeline_away_lgbm.predict(X_valid)

# Ensemble predictions (average of XGBoost and LGBM)
y_home_pred_ensemble = (y_home_pred_xgb + y_home_pred_lgbm) / 2
y_away_pred_ensemble = (y_away_pred_xgb + y_away_pred_lgbm) / 2

home_mae_ensemble = mean_absolute_error(y_home_valid, y_home_pred_ensemble)
home_rmse_ensemble = np.sqrt(mean_squared_error(y_home_valid, y_home_pred_ensemble))
away_mae_ensemble = mean_absolute_error(y_away_valid, y_away_pred_ensemble)
away_rmse_ensemble = np.sqrt(mean_squared_error(y_away_valid, y_away_pred_ensemble))

# Print evaluation metrics
print(f"\n--- Train1 Evaluation Results ---")
print(f"Classification Accuracy: {classification_accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Home Goals RMSE (Ensemble): {home_rmse_ensemble:.4f}")
print(f"Away Goals RMSE (Ensemble): {away_rmse_ensemble:.4f}")

# Check score-outcome consistency
rounded_home_goals = np.round(y_home_pred_ensemble)
rounded_away_goals = np.round(y_away_pred_ensemble)

# Convert score predictions to outcomes (0=away win, 1=draw, 2=home win)
score_outcomes = np.where(rounded_home_goals > rounded_away_goals, 2,
                         np.where(rounded_home_goals < rounded_away_goals, 0, 1))

consistency = accuracy_score(score_outcomes, y_class_pred)
print(f"Score-Outcome Consistency: {consistency:.4f}")

# Save the enhanced models
print("Saving trained models...")
joblib.dump(best_clf_pipeline, "train1_classifier.pkl")
joblib.dump((reg_pipeline_home, reg_pipeline_home_lgbm), "train1_regressor_home.pkl")
joblib.dump((reg_pipeline_away, reg_pipeline_away_lgbm), "train1_regressor_away.pkl")
eta_tracker.update(12, "Evaluation and saving models")

print("Train1 complete: Enhanced models trained on 70% of data and saved.")
