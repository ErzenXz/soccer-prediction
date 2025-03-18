#!/usr/bin/env python3
"""
Script to fetch today's matches and yesterday's results and store them in the Supabase database.
This can be run as a daily cron job to keep the database up to date.
"""

import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# API-Football Configuration
API_KEY = os.environ.get('API_FOOTBALL_KEY')
API_HOST = os.environ.get('API_FOOTBALL_HOST', 'v3.football.api-sports.io')
API_BASE_URL = f"https://{API_HOST}"

# Supabase Configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
supabase: Client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client created successfully")
    except Exception as e:
        print(f"Error creating Supabase client: {e}")
        sys.exit(1)
else:
    print("SUPABASE_URL or SUPABASE_KEY not set. Cannot proceed.")
    sys.exit(1)

def get_api_football_data(endpoint, params=None):
    """
    Make a request to the API-Football API
    """
    if not API_KEY:
        print("API_FOOTBALL_KEY not set. Cannot proceed.")
        sys.exit(1)
        
    headers = {
        'x-rapidapi-host': API_HOST,
        'x-rapidapi-key': API_KEY
    }

    url = f"{API_BASE_URL}/{endpoint}"
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        print(f"API Request successful: {endpoint} - {len(data.get('response', [])) if 'response' in data else 'N/A'} items")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error making API request to {endpoint}: {e}")
        return None

def store_matches_in_db(matches):
    """
    Store matches in the database
    """
    if not matches:
        return False
    
    try:
        count = 0
        # Store each match
        for match in matches:
            # First check if the match already exists
            existing_match = supabase.table('matches').select('id').eq('match_id', match['match_id']).execute()
            
            if existing_match.data and len(existing_match.data) > 0:
                print(f"Match {match['match_id']} already exists in database")
                continue
            
            # Prepare match data
            match_data = {
                'match_id': match['fixture']['id'],
                'date': match['fixture']['date'].split('T')[0],  # Extract YYYY-MM-DD
                'time': datetime.fromisoformat(match['fixture']['date'].replace('Z', '+00:00')).strftime('%H:%M'),
                'home_team': match['teams']['home']['name'],
                'away_team': match['teams']['away']['name'],
                'league': match['league']['name'],
                'country': match['league']['country'],
                'league_logo': match['league'].get('logo', ''),
                'home_logo': match['teams']['home'].get('logo', ''),
                'away_logo': match['teams']['away'].get('logo', '')
            }
            
            # Insert match
            supabase.table('matches').insert(match_data).execute()
            count += 1
        
        print(f"Successfully stored {count} new matches in database")
        return True
    except Exception as e:
        print(f"Error storing matches in database: {e}")
        return False

def store_results_in_db(results):
    """
    Store match results in the database
    """
    if not results:
        return False
    
    try:
        count = 0
        # Store each result
        for result in results:
            # Only process finished matches
            if result['fixture']['status']['short'] != 'FT':
                continue
                
            # First check if the result already exists
            existing_result = supabase.table('results').select('id').eq('match_id', result['fixture']['id']).execute()
            
            if existing_result.data and len(existing_result.data) > 0:
                print(f"Result for match {result['fixture']['id']} already exists in database")
                continue
            
            # Prepare result data
            result_data = {
                'match_id': result['fixture']['id'],
                'date': result['fixture']['date'].split('T')[0],  # Extract YYYY-MM-DD
                'home_team': result['teams']['home']['name'],
                'away_team': result['teams']['away']['name'],
                'home_score': result['goals']['home'],
                'away_score': result['goals']['away'],
                'league': result['league']['name'],
                'country': result['league']['country'],
                'league_logo': result['league'].get('logo', ''),
                'home_logo': result['teams']['home'].get('logo', ''),
                'away_logo': result['teams']['away'].get('logo', '')
            }
            
            # Insert result
            supabase.table('results').insert(result_data).execute()
            count += 1
        
        print(f"Successfully stored {count} new results in database")
        return True
    except Exception as e:
        print(f"Error storing results in database: {e}")
        return False

def update_today_matches():
    """
    Fetch and store today's matches
    """
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Fetching matches for today ({today})...")
    
    # Set up API parameters
    params = {
        'date': today,
    }
    
    # Make the API request
    response_data = get_api_football_data('fixtures', params)
    
    if not response_data or 'response' not in response_data:
        print("API request failed or returned invalid data.")
        return False
    
    # Store fixtures in database
    fixtures = response_data['response']
    return store_matches_in_db(fixtures)

def update_yesterday_results():
    """
    Fetch and store yesterday's results
    """
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Fetching results for yesterday ({yesterday})...")
    
    # Set up API parameters
    params = {
        'date': yesterday,
    }
    
    # Make the API request
    response_data = get_api_football_data('fixtures', params)
    
    if not response_data or 'response' not in response_data:
        print("API request failed or returned invalid data.")
        return False
    
    # Store results in database
    fixtures = response_data['response']
    return store_results_in_db(fixtures)

if __name__ == "__main__":
    print("Starting data update process...")
    
    # Update today's matches
    update_today_matches()
    
    # Update yesterday's results
    update_yesterday_results()
    
    print("Data update process completed.") 