
---
**Project:** Build a Flask web application that:  
- Fetches daily match data from [API-Football](https://www.api-football.com/).  
- Stores matches in a **Supabase database**.  
- Uses multiple AI models (Small, Medium, Large) to predict match results.  
- Updates daily with actual match results and evaluates each model’s accuracy.  
- Displays predictions, results, and model performance stats via a web interface.

---

### **Technology Stack**
- **Backend**: Flask (Python)  
- **Database**: Supabase (PostgreSQL)  
- **AI Models**: Pre-trained models (Small, Medium, Large)  
- **Scheduler**: Celery + Redis (or APScheduler) for periodic tasks  
- **Frontend**: Flask + Jinja2 or simple API for a separate frontend  

---

### **Key Features**
1. **Fetch Matches Daily**
   - Use API-Football to get all matches for the day.
   - Store these matches in Supabase.
   - Predict match results using all models.

2. **Store Results & Compare Model Accuracy**
   - The next day, fetch previous matches’ actual results.
   - Compare predictions vs. real outcomes.
   - Calculate and store accuracy per model.

3. **Daily Model Performance Dashboard**
   - Show how each model performed.
   - Display metrics: **correct outcome %, correct score %, avg goal error**.

---

### **Endpoints**
1. **`/predict-today`** (Triggers predictions for today’s matches)
2. **`/update-results`** (Fetches results & evaluates models)
3. **`/stats`** (Returns model performance stats)
4. **`/matches`** (Shows daily matches & predictions)

---

### **AI Agent Tasks**
1. **Setup Flask App & Supabase Database**
   - Define tables: `matches`, `predictions`, `results`, `model_stats`.

2. **Scheduled Tasks**
   - Fetch & store today’s matches at **00:00 UTC**.
   - Predict outcomes using all models.
   - Fetch yesterday’s results & evaluate model accuracy.

3. **Model Predictions**
   - Load trained models (small, medium, large, etc.).
   - Predict results for today’s matches.
   - Store predictions in Supabase.

4. **Evaluate Models**
   - Compare actual results vs. predictions.
   - Compute accuracy metrics: **Outcome %, Score %, Goal Error**.

5. **Create API Routes**
   - `/predict-today` (Trigger manually)
   - `/update-results` (Trigger manually)
   - `/stats` (Show model comparison)
   - `/matches` (Show today’s matches & predictions)

---

### **Example Supabase Tables**
#### **Matches Table**
| match_id | date | home_team | away_team | predicted | actual_result |
|----------|------|-----------|-----------|-----------|--------------|
| 1        | 2025-03-10 | Barcelona | Real Madrid | 2-1 | 1-1 |

#### **Predictions Table**
| match_id | model_name | predicted_score | predicted_outcome |
|----------|-----------|----------------|------------------|
| 1        | Small V1  | 2-1           | Home Win        |
| 1        | Medium V2 | 1-1           | Draw            |

#### **Model Stats Table**
| model_name | correct_outcome % | correct_score % | avg_goal_error |
|------------|------------------|----------------|---------------|
| Small V1   | 55%              | 20%            | 0.9           |
| Large V2   | 70%              | 35%            | 0.5           |

---

### **Deployment**
- Run on **Gunicorn** with **NGINX + Flask**.
- Use **Celery + Redis** to schedule tasks.
- Expose API endpoints for frontend integration.

---