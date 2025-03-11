# train-small.py - Training script with 30% data sample and progress tracking
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from lightgbm import LGBMClassifier, LGBMRegressor
import joblib
import time
from tqdm import tqdm
from datetime import datetime, timedelta

# Function to estimate and display ETA
class ETATracker:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.start_time = time.time()
        self.step_times = []
    
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


print("Starting training script with 30% data sample...")
total_steps = 10  # Will be updated after data loading
eta_tracker = ETATracker(total_steps)

# Step 1: Load and sample data
print("Loading data...")
start_time = time.time()
data = pd.read_parquet("./data/games.parquet")

# Sample 30% of the data for faster training
data = data.sample(frac=0.3, random_state=42)
print(f"Using {len(data)} matches for training (30% sample)")
eta_tracker.update(1, "Data loading")

# Step 2: Feature engineering - basic temporal features
print("Feature engineering...")
data["result"] = data.apply(lambda row: 1 if row["gh"] > row["ga"]
                          else (-1 if row["gh"] < row["ga"] else 0), axis=1)
data["home_goals"] = data["gh"]
data["away_goals"] = data["ga"]

data["date"] = pd.to_datetime(data["date"])
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day_of_week"] = data["date"].dt.dayofweek
data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

# Sort data chronologically (needed for form features)
data = data.sort_values("date")
eta_tracker.update(2, "Basic feature engineering")

# Step 3: Add team form features with progress bar
print("Calculating team form...")
def calculate_team_form(df, team_col, result_col, window=3):
    teams = df[team_col].unique()
    form_dict = {}
    
    # Use tqdm for inner loop progress
    for team in tqdm(teams, desc=f"{team_col} form", leave=False):
        team_matches = df[df[team_col] == team].copy()
        if team_col == 'home':
            team_matches['result_adj'] = team_matches[result_col]
        else:
            team_matches['result_adj'] = -team_matches[result_col]
        
        team_matches['rolling_form'] = team_matches['result_adj'].rolling(window, min_periods=1).mean()
        form_dict[team] = dict(zip(team_matches['date'], team_matches['rolling_form']))
    
    return form_dict

# Calculate only basic form metrics for speed
home_forms = calculate_team_form(data, 'home', 'result', window=3)
away_forms = calculate_team_form(data, 'away', 'result', window=3)

data['home_form'] = data.apply(lambda x: home_forms.get(x['home'], {}).get(x['date'], 0), axis=1)
data['away_form'] = data.apply(lambda x: away_forms.get(x['away'], {}).get(x['date'], 0), axis=1)
eta_tracker.update(3, "Team form calculation")

# Step 4: Prepare feature matrix
features = [
    "year", "month", "day_of_week", "is_weekend",
    "full_time", "competition",
    "home", "away",
    "home_country", "away_country",
    "level",
    "home_form", "away_form"
]

print("Preparing features and targets...")
X = data[features]
y_class = data["result"].map({-1: 0, 0: 1, 1: 2})  # Transform to 0,1,2 for XGBoost
y_home = data["home_goals"]
y_away = data["away_goals"]

# Split data for model evaluation (80% train, 20% validation)
X_train, X_valid, y_class_train, y_class_valid, y_home_train, y_home_valid, y_away_train, y_away_valid = train_test_split(
    X, y_class, y_home, y_away, test_size=0.2, random_state=42, stratify=y_class)
eta_tracker.update(4, "Feature preparation")

# Step 5: Create preprocessing pipeline
categorical_features = ["full_time", "competition", "home", "away", "home_country", "away_country", "level"]
numerical_features = [col for col in features if col not in categorical_features]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numerical_features)
])
eta_tracker.update(5, "Preprocessing setup")

# Step 6: Train classification model
print("Training classification model...")
clf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb.XGBClassifier(
        objective="multi:softmax", 
        num_class=3, 
        n_estimators=100,  # Reduced for speed
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ))
])

clf_pipeline.fit(X_train, y_class_train)
eta_tracker.update(6, "Classification model")

# Step 7: Train home goals regression
print("Training home goals regression...")
reg_pipeline_home = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,  # Reduced for speed
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ))
])
reg_pipeline_home.fit(X_train, y_home_train)
eta_tracker.update(7, "Home goals model")

# Step 8: Train away goals regression
print("Training away goals regression...")
reg_pipeline_away = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,  # Reduced for speed
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ))
])
reg_pipeline_away.fit(X_train, y_away_train)
eta_tracker.update(8, "Away goals model")

# Step 9: Evaluate models
print("\nEvaluating models...")
# Classification metrics
y_class_pred = clf_pipeline.predict(X_valid)
classification_accuracy = accuracy_score(y_class_valid, y_class_pred)

# Regression metrics
y_home_pred = reg_pipeline_home.predict(X_valid)
y_away_pred = reg_pipeline_away.predict(X_valid)
home_rmse = np.sqrt(mean_squared_error(y_home_valid, y_home_pred))
away_rmse = np.sqrt(mean_squared_error(y_away_valid, y_away_pred))

# Check score-outcome consistency
rounded_home_goals = np.round(y_home_pred)
rounded_away_goals = np.round(y_away_pred)
score_outcomes = np.where(rounded_home_goals > rounded_away_goals, 2,
                         np.where(rounded_home_goals < rounded_away_goals, 0, 1))
consistency = accuracy_score(score_outcomes, y_class_pred)

print(f"\n--- Evaluation Results ---")
print(f"Classification Accuracy: {classification_accuracy:.4f}")
print(f"Home Goals RMSE: {home_rmse:.4f}")
print(f"Away Goals RMSE: {away_rmse:.4f}")
print(f"Score-Outcome Consistency: {consistency:.4f}")
eta_tracker.update(9, "Model evaluation")

# Step 10: Save models
print("Saving models...")
joblib.dump(clf_pipeline, "train_small_classifier.pkl")
joblib.dump(reg_pipeline_home, "train_small_regressor_home.pkl")
joblib.dump(reg_pipeline_away, "train_small_regressor_away.pkl")
eta_tracker.update(10, "Saving models")

# Show some sample predictions
print("\n=== Sample Predictions ===")
sample_indices = np.random.choice(len(X_valid), 5, replace=False)
sample_X = X_valid.iloc[sample_indices]
sample_teams = pd.DataFrame({
    'Home': sample_X['home'],
    'Away': sample_X['away'],
    'Actual Result': [
        "Home Win" if r == 2 else "Draw" if r == 1 else "Away Win" 
        for r in y_class_valid.iloc[sample_indices]
    ],
    'Predicted Result': [
        "Home Win" if r == 2 else "Draw" if r == 1 else "Away Win" 
        for r in clf_pipeline.predict(sample_X)
    ],
    'Actual Score': [
        f"{h} - {a}" for h, a in zip(
            y_home_valid.iloc[sample_indices], y_away_valid.iloc[sample_indices]
        )
    ],
    'Predicted Score': [
        f"{round(h)} - {round(a)}" for h, a in zip(
            reg_pipeline_home.predict(sample_X), reg_pipeline_away.predict(sample_X)
        )
    ]
})

print(sample_teams.to_string(index=False))
print("\nTrain-small complete: Models trained on 30% data sample and saved.")
