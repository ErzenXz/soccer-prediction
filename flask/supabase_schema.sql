-- Create matches table
CREATE TABLE IF NOT EXISTS matches (
    id SERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL UNIQUE,
    date DATE NOT NULL,
    time VARCHAR(10) NOT NULL,
    home_team VARCHAR(255) NOT NULL,
    away_team VARCHAR(255) NOT NULL,
    league VARCHAR(255) NOT NULL,
    country VARCHAR(255),
    league_logo TEXT,
    home_logo TEXT,
    away_logo TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Add index for faster date-based queries
    CONSTRAINT idx_matches_date UNIQUE (date, match_id)
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    model_name VARCHAR(50) NOT NULL,
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    outcome VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Each match should have only one prediction per model
    CONSTRAINT idx_predictions_unique UNIQUE (match_id, model_name)
);

-- Create results table
CREATE TABLE IF NOT EXISTS results (
    id SERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL UNIQUE,
    date DATE NOT NULL,
    home_team VARCHAR(255) NOT NULL,
    away_team VARCHAR(255) NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    league VARCHAR(255) NOT NULL,
    country VARCHAR(255),
    league_logo TEXT,
    home_logo TEXT,
    away_logo TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Add index for faster date-based queries
    CONSTRAINT idx_results_date UNIQUE (date, match_id)
);

-- Create model_stats table
CREATE TABLE IF NOT EXISTS model_stats (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    correct_outcome_percentage FLOAT NOT NULL,
    correct_score_percentage FLOAT NOT NULL,
    avg_goal_error FLOAT NOT NULL,
    matches_predicted INTEGER NOT NULL,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Each model should have only one stats record
    CONSTRAINT idx_model_stats_unique UNIQUE (model_name)
);

-- Add RLS policies to allow authenticated users to read but only service role to write
ALTER TABLE matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE results ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_stats ENABLE ROW LEVEL SECURITY;

-- Create policies for read access
CREATE POLICY "Allow public read access to matches" 
    ON matches FOR SELECT USING (true);

CREATE POLICY "Allow public read access to predictions" 
    ON predictions FOR SELECT USING (true);

CREATE POLICY "Allow public read access to results" 
    ON results FOR SELECT USING (true);

CREATE POLICY "Allow public read access to model_stats" 
    ON model_stats FOR SELECT USING (true);

-- Comments for tables
COMMENT ON TABLE matches IS 'Soccer matches fetched from API-Football';
COMMENT ON TABLE predictions IS 'Predictions made by AI models for matches';
COMMENT ON TABLE results IS 'Actual results of matches';
COMMENT ON TABLE model_stats IS 'Performance statistics for AI prediction models'; 