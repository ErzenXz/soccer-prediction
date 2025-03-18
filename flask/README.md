# Soccer Prediction Web Application

A Flask web application that fetches soccer match data from API-Football, uses AI models to predict results, and displays predictions and model performance statistics.

## Features

- View predictions for today's matches and navigate to other dates
- Browse real match data and results from API-Football
- Compare predictions from multiple AI models (Small, Medium, Large)
- Track and display model performance statistics
- Daily updates of match results and model evaluations
- **Database caching** to reduce API calls and stay within API-Football's request limits

## Technology Stack

- **Backend**: Flask (Python)
- **API**: [API-Football](https://www.api-football.com/) for real match data
- **Database**: PostgreSQL (via Supabase or any PostgreSQL provider)
- **AI Models**: Multiple prediction models
- **Scheduler**: Celery + Redis for periodic tasks
- **Frontend**: Bootstrap 5 with responsive design

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Copy `.env.example` to `.env` and fill in your API credentials:
   ```bash
   cp .env.example .env
   ```

## API-Football Setup

1. Sign up for an account at [API-Football](https://www.api-football.com/)
2. Subscribe to a plan (they offer free tier with limited requests)
3. Get your API key from your dashboard
4. Add your API key to the `.env` file:
   ```
   API_FOOTBALL_KEY=your_api_key_here
   ```

## PostgreSQL Database Setup

1. Create a PostgreSQL database (you can use Supabase, ElephantSQL, or any other PostgreSQL provider)
2. Get your PostgreSQL connection string
3. Add it to your `.env` file:
   ```
   DATABASE_URL=postgresql://username:password@host:port/database
   ```
4. Set up the database schema by executing the SQL in `supabase_schema.sql` in the PostgreSQL database

### If You're Using Supabase:

1. Create a free account at [Supabase](https://supabase.com/) if you don't have one
2. Create a new project
3. Get your PostgreSQL connection string from Project Settings > Database > Connection string > URI
4. This connection uses the `postgres` user which has full privileges, so you won't have RLS policy issues

### Database Schema

The application uses the following tables:

1. **matches** - Stores match information
2. **predictions** - Stores model predictions for matches
3. **results** - Stores actual match results
4. **model_stats** - Stores performance statistics for each model

## How the Data Flow Works

1. When a user requests matches for a specific date:

   - First, check if the data exists in the PostgreSQL database
   - If found, return the data from the database
   - If not found, fetch from API-Football and store in the database for future use

2. This approach:
   - Reduces the number of API calls to stay within API-Football limits
   - Improves response times for repeat queries
   - Builds a historical database of matches and predictions

## Running the Application

### Development Mode

```bash
flask run
```

Visit `http://localhost:5000` in your browser.

### Production Mode

```bash
gunicorn app:app
```

## Diagnostic Tools

The application includes diagnostic tools to help troubleshoot issues:

1. **`/db-test`** - Tests the database connection and operations
2. **`/fetch-and-store`** - Manually fetch and store matches for a specific date

## Scheduled Tasks

The application includes scheduled tasks to:

- Fetch today's matches at midnight UTC
- Generate predictions for all matches
- Update results and evaluate model performance

To run the Celery worker:

```bash
celery -A app.celery worker --loglevel=info
```

To run the Celery beat scheduler:

```bash
celery -A app.celery beat --loglevel=info
```

## API Endpoints

- **`/`** - Home page, redirects to today's matches
- **`/matches`** - View matches and predictions for a specific date
- **`/results`** - View match results for a specific date
- **`/stats`** - View AI model performance statistics
- **`/predict-today`** - Trigger predictions for today's matches (POST)
- **`/update-results`** - Update results and evaluate models (POST)

## License

MIT
