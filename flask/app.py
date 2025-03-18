from flask import Flask, render_template, request, redirect, url_for, jsonify
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import requests
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Add current_year to all templates
@app.context_processor
def inject_current_year():
    return {'current_year': datetime.now().year}

# API-Football Configuration
API_KEY = os.environ.get('API_FOOTBALL_KEY')
API_HOST = os.environ.get('API_FOOTBALL_HOST', 'v3.football.api-sports.io')
API_BASE_URL = f"https://{API_HOST}"

# PostgreSQL Configuration (using Supabase PostgreSQL)
DB_URL = os.environ.get('DATABASE_URL')
db_connection = None

def get_db_connection():
    """Get a PostgreSQL database connection"""
    global db_connection
    if db_connection is None:
        try:
            logger.info("Creating new database connection...")
            db_connection = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
            db_connection.autocommit = True
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return None
    return db_connection

def close_db_connection():
    """Close the PostgreSQL database connection"""
    global db_connection
    if db_connection is not None:
        db_connection.close()
        db_connection = None
        logger.info("Database connection closed")

# Top leagues to prioritize in display (league IDs from API-Football)
TOP_LEAGUES = [
    39,  # Premier League
    140, # La Liga
    135, # Serie A
    78,  # Bundesliga
    61,  # Ligue 1
    2,   # Champions League
    3,   # Europa League
]

def get_api_football_data(endpoint, params=None):
    """
    Make a request to the API-Football API
    """
    headers = {
        'x-rapidapi-host': API_HOST,
        'x-rapidapi-key': API_KEY
    }

    url = f"{API_BASE_URL}/{endpoint}"
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        logger.info(f"API Request successful: {endpoint} - {len(data.get('response', [])) if 'response' in data else 'N/A'} items")
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making API request to {endpoint}: {e}")
        return None

def check_db_for_matches(date_str):
    """
    Check if matches for a specific date exist in the database
    """
    conn = get_db_connection()
    if conn is None:
        logger.warning("Database connection failed, cannot check database")
        return None
    
    try:
        logger.info(f"Checking database for matches on {date_str}")
        with conn.cursor() as cur:
            # Query the matches table for the given date
            cur.execute("SELECT * FROM matches WHERE date = %s", (date_str,))
            matches = cur.fetchall()
            
            if matches and len(matches) > 0:
                logger.info(f"Found {len(matches)} matches in database for {date_str}")
                
                # Get predictions for these matches
                match_ids = [match['id'] for match in matches]
                
                predictions = []
                if match_ids:
                    placeholders = ','.join(['%s'] * len(match_ids))
                    cur.execute(f"SELECT * FROM predictions WHERE match_id IN ({placeholders})", tuple(match_ids))
                    predictions = cur.fetchall()
                    
                logger.info(f"Found {len(predictions)} predictions in database")
                
                # Organize predictions by match_id
                predictions_by_match = {}
                for prediction in predictions:
                    match_id = prediction['match_id']
                    if match_id not in predictions_by_match:
                        predictions_by_match[match_id] = {}
                    
                    model_name = prediction['model_name']
                    predictions_by_match[match_id][model_name] = {
                        "home_score": prediction['home_score'],
                        "away_score": prediction['away_score'],
                        "outcome": prediction['outcome']
                    }
                
                # Add predictions to each match
                formatted_matches = []
                for match in matches:
                    match_id = match['id']
                    api_match_id = match['match_id']  # This is the ID from the API
                    formatted_match = {
                        "match_id": api_match_id,  # Use the API match_id for consistency
                        "db_id": match_id,         # Also store the DB ID for reference
                        "date": match['date'].strftime('%Y-%m-%d'),
                        "time": match['time'],
                        "home_team": match['home_team'],
                        "away_team": match['away_team'],
                        "league": match['league'],
                        "country": match['country'],
                        "league_logo": match['league_logo'],
                        "home_logo": match['home_logo'],
                        "away_logo": match['away_logo'],
                        "predictions": predictions_by_match.get(match_id, generate_mock_predictions(match['home_team'], match['away_team']))
                    }
                    formatted_matches.append(formatted_match)
                
                return formatted_matches
            
            logger.info(f"No matches found in database for {date_str}")
            return None
    except Exception as e:
        logger.error(f"Error checking database for matches: {e}")
        return None

def store_matches_in_db(matches):
    """
    Store matches and their predictions in the database
    """
    conn = get_db_connection()
    if conn is None or not matches:
        logger.warning("Cannot store matches: Database connection failed or no matches provided")
        return False
    
    try:
        stored_count = 0
        logger.info(f"Attempting to store {len(matches)} matches in database")
        
        with conn.cursor() as cur:
            # Store each match
            for match in matches:
                # The match_id we need to check is the API match_id
                api_match_id = match['match_id']
                
                # First check if the match already exists
                cur.execute("SELECT id FROM matches WHERE match_id = %s", (api_match_id,))
                existing_match = cur.fetchone()
                
                if existing_match:
                    logger.info(f"Match {api_match_id} already exists in database")
                    continue
                
                # Prepare match data
                match_data = (
                    api_match_id,
                    match['date'],
                    match['time'],
                    match['home_team'],
                    match['away_team'],
                    match['league'],
                    match.get('country', ''),
                    match.get('league_logo', ''),
                    match.get('home_logo', ''),
                    match.get('away_logo', '')
                )
                
                # Insert match
                logger.info(f"Inserting match {api_match_id} into database")
                cur.execute("""
                    INSERT INTO matches 
                    (match_id, date, time, home_team, away_team, league, country, league_logo, home_logo, away_logo) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                """, match_data)
                
                db_match_id = cur.fetchone()['id']
                stored_count += 1
                logger.info(f"Match inserted with database ID: {db_match_id}")
                
                # Store predictions
                for model_name, prediction in match['predictions'].items():
                    prediction_data = (
                        db_match_id,
                        model_name,
                        prediction['home_score'],
                        prediction['away_score'],
                        prediction['outcome']
                    )
                    
                    logger.info(f"Inserting prediction for match {db_match_id}, model {model_name}")
                    cur.execute("""
                        INSERT INTO predictions 
                        (match_id, model_name, home_score, away_score, outcome) 
                        VALUES (%s, %s, %s, %s, %s)
                    """, prediction_data)
        
        logger.info(f"Successfully stored {stored_count} matches in database")
        return True
    except Exception as e:
        logger.error(f"Error storing matches in database: {e}")
        return False

def check_db_for_results(date_str):
    """
    Check if results for a specific date exist in the database
    """
    conn = get_db_connection()
    if conn is None:
        logger.warning("Database connection failed, cannot check database")
        return None
    
    try:
        logger.info(f"Checking database for results on {date_str}")
        with conn.cursor() as cur:
            # Query the results table for the given date
            cur.execute("SELECT * FROM results WHERE date = %s", (date_str,))
            results = cur.fetchall()
            
            if results and len(results) > 0:
                logger.info(f"Found {len(results)} results in database for {date_str}")
                
                # Format the date in each result
                formatted_results = []
                for result in results:
                    result_copy = dict(result)
                    result_copy['date'] = result['date'].strftime('%Y-%m-%d')
                    formatted_results.append(result_copy)
                
                return formatted_results
            
            logger.info(f"No results found in database for {date_str}")
            return None
    except Exception as e:
        logger.error(f"Error checking database for results: {e}")
        return None

def store_results_in_db(results):
    """
    Store match results in the database
    """
    conn = get_db_connection()
    if conn is None or not results:
        logger.warning("Cannot store results: Database connection failed or no results provided")
        return False
    
    try:
        stored_count = 0
        logger.info(f"Attempting to store {len(results)} results in database")
        
        with conn.cursor() as cur:
            # Store each result
            for result in results:
                # The match_id we need to check is the API match_id
                api_match_id = result['match_id']
                
                # First check if the result already exists
                cur.execute("SELECT id FROM results WHERE match_id = %s", (api_match_id,))
                existing_result = cur.fetchone()
                
                if existing_result:
                    logger.info(f"Result for match {api_match_id} already exists in database")
                    continue
                
                # Prepare result data
                result_data = (
                    api_match_id,
                    result['date'],
                    result['home_team'],
                    result['away_team'],
                    result['home_score'],
                    result['away_score'],
                    result['league'],
                    result.get('country', ''),
                    result.get('league_logo', ''),
                    result.get('home_logo', ''),
                    result.get('away_logo', '')
                )
                
                # Insert result
                logger.info(f"Inserting result for match {api_match_id} into database")
                cur.execute("""
                    INSERT INTO results 
                    (match_id, date, home_team, away_team, home_score, away_score, league, country, league_logo, home_logo, away_logo) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, result_data)
                
                stored_count += 1
        
        logger.info(f"Successfully stored {stored_count} results in database")
        return True
    except Exception as e:
        logger.error(f"Error storing results in database: {e}")
        return False

def create_match_object(fixture, date_str):
    """
    Create a match object from a fixture object
    """
    # Extract match time
    try:
        match_time = datetime.fromisoformat(fixture['fixture']['date'].replace('Z', '+00:00')).strftime('%H:%M')
    except (ValueError, TypeError):
        match_time = "00:00"  # Default time if there's an error
    
    # Create match object
    match = {
        "match_id": fixture['fixture']['id'],
        "date": date_str,
        "time": match_time,
        "home_team": fixture['teams']['home']['name'],
        "away_team": fixture['teams']['away']['name'],
        "league": fixture['league']['name'],
        "country": fixture['league']['country'],
        "league_logo": fixture['league'].get('logo', ''),
        "home_logo": fixture['teams']['home'].get('logo', ''),
        "away_logo": fixture['teams']['away'].get('logo', ''),
        "predictions": generate_mock_predictions(fixture['teams']['home']['name'], fixture['teams']['away']['name'])
    }
    return match

def generate_mock_predictions(home_team, away_team):
    """
    Generate mock predictions for a match.
    In a real application, this would call actual prediction models.
    """
    import random
    
    # Generate more realistic scores based on team strength
    # For now, just random scores
    home_strong = random.random() > 0.4
    away_strong = random.random() > 0.4
    
    home_score_small = random.randint(0, 2) + (1 if home_strong else 0)
    away_score_small = random.randint(0, 2) + (1 if away_strong else 0)
    
    home_score_medium = random.randint(0, 2) + (1 if home_strong else 0)
    away_score_medium = random.randint(0, 2) + (1 if away_strong else 0)
    
    home_score_large = random.randint(0, 2) + (1 if home_strong else 0)
    away_score_large = random.randint(0, 2) + (1 if away_strong else 0)
    
    # Determine outcomes
    small_outcome = "HOME_WIN" if home_score_small > away_score_small else "AWAY_WIN" if away_score_small > home_score_small else "DRAW"
    medium_outcome = "HOME_WIN" if home_score_medium > away_score_medium else "AWAY_WIN" if away_score_medium > home_score_medium else "DRAW"
    large_outcome = "HOME_WIN" if home_score_large > away_score_large else "AWAY_WIN" if away_score_large > home_score_large else "DRAW"
    
    return {
        "small_model": {
            "home_score": home_score_small, 
            "away_score": away_score_small, 
            "outcome": small_outcome
        },
        "medium_model": {
            "home_score": home_score_medium, 
            "away_score": away_score_medium, 
            "outcome": medium_outcome
        },
        "large_model": {
            "home_score": home_score_large, 
            "away_score": away_score_large, 
            "outcome": large_outcome
        }
    }

# Mock data for development - used as fallback if API is not available
def get_mock_matches(date_str=None):
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Mock matches data
    return [
        {
            "match_id": 1,
            "date": date_str,
            "time": "15:00",
            "home_team": "Barcelona",
            "away_team": "Real Madrid",
            "league": "La Liga",
            "country": "Spain",
            "league_logo": "https://media.api-sports.io/football/leagues/140.png",
            "home_logo": "https://media.api-sports.io/football/teams/529.png",
            "away_logo": "https://media.api-sports.io/football/teams/541.png",
            "predictions": {
                "small_model": {"home_score": 2, "away_score": 1, "outcome": "HOME_WIN"},
                "medium_model": {"home_score": 1, "away_score": 1, "outcome": "DRAW"},
                "large_model": {"home_score": 3, "away_score": 1, "outcome": "HOME_WIN"}
            }
        },
        {
            "match_id": 2,
            "date": date_str,
            "time": "17:30",
            "home_team": "Manchester City",
            "away_team": "Liverpool",
            "league": "Premier League",
            "country": "England",
            "league_logo": "https://media.api-sports.io/football/leagues/39.png",
            "home_logo": "https://media.api-sports.io/football/teams/50.png",
            "away_logo": "https://media.api-sports.io/football/teams/40.png",
            "predictions": {
                "small_model": {"home_score": 2, "away_score": 2, "outcome": "DRAW"},
                "medium_model": {"home_score": 3, "away_score": 1, "outcome": "HOME_WIN"},
                "large_model": {"home_score": 2, "away_score": 1, "outcome": "HOME_WIN"}
            }
        }
    ]

def get_matches_for_date(date_str):
    """
    Get matches for a specific date. First check database, then fall back to API.
    """
    # Check if matches exist in the database first
    db_matches = check_db_for_matches(date_str)
    if db_matches:
        return db_matches
    
    logger.info(f"No matches found in database for {date_str}, fetching from API...")
    
    # If not in database, check API-Football
    # Check if API key is set
    if not API_KEY:
        logger.warning("API_FOOTBALL_KEY not set. Using mock data.")
        return get_mock_matches(date_str)
    
    # Set up API parameters
    params = {
        'date': date_str,
    }
    
    # Make the API request
    response_data = get_api_football_data('fixtures', params)
    
    # If API request failed, fall back to mock data
    if not response_data or 'response' not in response_data:
        logger.warning("API request failed or returned invalid data. Using mock data.")
        return get_mock_matches(date_str)
    
    # Parse the response
    matches = []
    fixtures = response_data['response']
    
    # First add top league matches
    for fixture in fixtures:
        if fixture['league']['id'] in TOP_LEAGUES:
            match = create_match_object(fixture, date_str)
            matches.append(match)
    
    # Then add other leagues if we don't have enough matches
    if len(matches) < 10:
        for fixture in fixtures:
            if fixture['league']['id'] not in TOP_LEAGUES:
                # Skip if already added from top leagues
                if any(m['match_id'] == fixture['fixture']['id'] for m in matches):
                    continue
                    
                match = create_match_object(fixture, date_str)
                matches.append(match)
                
                # Stop after reaching 15 matches total
                if len(matches) >= 15:
                    break
    
    # Sort matches by time
    matches.sort(key=lambda x: x['time'])
    
    # If no matches found, use mock data
    if not matches:
        logger.warning(f"No matches found for {date_str}. Using mock data instead.")
        return get_mock_matches(date_str)
    
    # Store matches in database for future use
    store_matches_in_db(matches)
        
    return matches

def get_match_results(date_str):
    """
    Get match results for a specific date. First check database, then fall back to API.
    """
    # Check if results exist in the database first
    db_results = check_db_for_results(date_str)
    if db_results:
        return db_results
    
    logger.info(f"No results found in database for {date_str}, fetching from API...")
    
    # If not in database, check API-Football
    # Check if API key is set
    if not API_KEY:
        logger.warning("API_FOOTBALL_KEY not set. Using mock data.")
        return []
    
    # Set up API parameters
    params = {
        'date': date_str,
    }
    
    # Make the API request
    response_data = get_api_football_data('fixtures', params)
    
    # If API request failed, return empty list
    if not response_data or 'response' not in response_data:
        logger.warning("API request failed or returned invalid data.")
        return []
    
    # Parse the response
    results = []
    for fixture in response_data['response']:
        # Only include matches that have finished
        if fixture['fixture']['status']['short'] == 'FT':
            result = {
                "match_id": fixture['fixture']['id'],
                "date": date_str,
                "home_team": fixture['teams']['home']['name'],
                "away_team": fixture['teams']['away']['name'],
                "home_score": fixture['goals']['home'],
                "away_score": fixture['goals']['away'],
                "league": fixture['league']['name'],
                "country": fixture['league']['country'],
                "league_logo": fixture['league'].get('logo', ''),
                "home_logo": fixture['teams']['home'].get('logo', ''),
                "away_logo": fixture['teams']['away'].get('logo', '')
            }
            results.append(result)
    
    # Sort results by league (prioritizing top leagues)
    def league_priority(result):
        league_id = next((l_id for l_id in TOP_LEAGUES if result.get('league_id') == l_id), 999)
        return (league_id, result['league'], result.get('time', ''))
    
    results.sort(key=league_priority)
    
    # Store results in database for future use
    store_results_in_db(results)
    
    return results

def get_mock_model_stats():
    return {
        "small_model": {
            "correct_outcome_percentage": 55,
            "correct_score_percentage": 20,
            "avg_goal_error": 0.9,
            "matches_predicted": 100
        },
        "medium_model": {
            "correct_outcome_percentage": 62,
            "correct_score_percentage": 28,
            "avg_goal_error": 0.7,
            "matches_predicted": 100
        },
        "large_model": {
            "correct_outcome_percentage": 70,
            "correct_score_percentage": 35,
            "avg_goal_error": 0.5,
            "matches_predicted": 100
        }
    }

@app.route('/')
def index():
    today = datetime.now().strftime("%Y-%m-%d")
    return redirect(url_for('matches', date=today))

@app.route('/matches')
def matches():
    # Get date from query parameter, default to today
    date_param = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    
    # Special case for May 15th as per the request
    if date_param == "2024-05-15":
        logger.info("Special case: Requesting matches for May 15, 2024")
    
    matches = get_matches_for_date(date_param)
    
    # Generate previous and next day links
    date_obj = datetime.strptime(date_param, "%Y-%m-%d")
    prev_day = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")
    next_day = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
    
    return render_template('matches.html', 
                          matches=matches, 
                          date=date_param, 
                          prev_day=prev_day, 
                          next_day=next_day,
                          formatted_date=date_obj.strftime("%A, %B %d, %Y"))

@app.route('/stats')
def stats():
    model_stats = get_mock_model_stats()
    return render_template('stats.html', stats=model_stats)

@app.route('/results')
def results():
    # Get date from query parameter, default to yesterday
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    date_param = request.args.get('date', yesterday)
    
    # Get match results
    results = get_match_results(date_param)
    
    # Generate previous and next day links
    date_obj = datetime.strptime(date_param, "%Y-%m-%d")
    prev_day = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")
    next_day = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
    
    return render_template('results.html', 
                          results=results, 
                          date=date_param, 
                          prev_day=prev_day, 
                          next_day=next_day,
                          formatted_date=date_obj.strftime("%A, %B %d, %Y"))

@app.route('/predict-today', methods=['POST'])
def predict_today():
    # This would trigger the prediction process for today's matches
    # For now, just return success message
    return jsonify({"status": "success", "message": "Predictions for today's matches have been generated"})

@app.route('/update-results', methods=['POST'])
def update_results():
    # This would update results and evaluate models
    # For now, just return success message
    return jsonify({"status": "success", "message": "Results updated and models evaluated"})

@app.route('/db-test')
def db_test():
    """
    Diagnostic route to test database connection and operations
    """
    results = {}
    
    # Test 1: Check database connection
    conn = get_db_connection()
    if conn:
        results["connection"] = "Connected to PostgreSQL database"
    else:
        results["connection"] = "NOT CONNECTED to database. Check your DATABASE_URL environment variable."
        return jsonify({"status": "error", "tests": results})
    
    # Test 2: Check table existence by trying to count records
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM matches")
            count = cur.fetchone()['count']
            results["matches_table"] = f"OK - {count} records"
            
            cur.execute("SELECT COUNT(*) FROM predictions")
            count = cur.fetchone()['count']
            results["predictions_table"] = f"OK - {count} records"
            
            cur.execute("SELECT COUNT(*) FROM results")
            count = cur.fetchone()['count']
            results["results_table"] = f"OK - {count} records"
    except Exception as e:
        results["table_check"] = f"Error: {str(e)}"
        return jsonify({"status": "error", "tests": results})
    
    # Test 3: Try to insert a test record (and then delete it)
    try:
        with conn.cursor() as cur:
            test_data = (
                999999,  # Using a high number unlikely to conflict
                datetime.now().strftime("%Y-%m-%d"),
                datetime.now().strftime("%H:%M"),
                'Test Home',
                'Test Away',
                'Test League',
                'Test Country',
                '',
                '',
                ''
            )
            
            # Insert test record
            cur.execute("""
                INSERT INTO matches 
                (match_id, date, time, home_team, away_team, league, country, league_logo, home_logo, away_logo) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """, test_data)
            
            test_id = cur.fetchone()['id']
            results["insert_test"] = f"OK - Inserted test record with ID {test_id}"
            
            # Delete the test record
            cur.execute("DELETE FROM matches WHERE id = %s", (test_id,))
            results["delete_test"] = "OK - Deleted test record"
    except Exception as e:
        results["write_test"] = f"Error: {str(e)}"
    
    # Test 4: Attempt to run a simple SQL query
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM matches LIMIT 1")
            results["sql_query"] = "OK - SQL query executed successfully"
    except Exception as e:
        results["sql_query"] = f"Error: {str(e)}"
    
    # Add environment variables (masked for security)
    env_vars = {
        "DATABASE_URL": "***" if DB_URL else "Not set",
        "API_FOOTBALL_KEY": "***" if API_KEY else "Not set"
    }
    
    return jsonify({
        "status": "success",
        "tests": results,
        "environment": env_vars
    })

@app.route('/fetch-and-store', methods=['GET', 'POST'])
def fetch_and_store():
    """
    Route to manually fetch and store matches for a specific date
    """
    if request.method == 'POST':
        date_str = request.form.get('date')
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Force fetch from API regardless of DB state
        logger.info(f"Manually fetching matches for {date_str} from API")
        
        if not API_KEY:
            return jsonify({"status": "error", "message": "API_FOOTBALL_KEY not set"})
        
        # Set up API parameters
        params = {
            'date': date_str,
        }
        
        # Make the API request
        response_data = get_api_football_data('fixtures', params)
        
        if not response_data or 'response' not in response_data:
            return jsonify({"status": "error", "message": "API request failed or returned invalid data"})
        
        # Parse fixtures and create match objects
        fixtures = response_data['response']
        matches = []
        
        for fixture in fixtures:
            match = create_match_object(fixture, date_str)
            matches.append(match)
        
        # Store matches in database
        result = store_matches_in_db(matches)
        
        if result:
            return jsonify({
                "status": "success", 
                "message": f"Successfully fetched and stored {len(matches)} matches for {date_str}",
                "matches_count": len(matches)
            })
        else:
            return jsonify({"status": "error", "message": "Failed to store matches in database"})
    
    # If GET request, return a simple form to select the date
    return '''
        <form method="post">
            <h2>Manually fetch and store matches</h2>
            <p>Enter date (YYYY-MM-DD):</p>
            <input type="date" name="date" required>
            <button type="submit">Fetch and Store</button>
        </form>
    '''

@app.teardown_appcontext
def teardown_db(exception):
    close_db_connection()

if __name__ == '__main__':
    app.run(debug=True) 