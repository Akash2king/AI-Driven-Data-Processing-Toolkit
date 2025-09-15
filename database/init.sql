-- Initialize the survey_cleaner database
-- This script will be run automatically when the PostgreSQL container starts

-- Create the database if it doesn't exist
CREATE DATABASE survey_cleaner;

-- Connect to the survey_cleaner database
\c survey_cleaner;

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create initial admin user (optional)
-- You can modify this as needed
DO $$ 
BEGIN
    -- Additional initialization code can go here
    RAISE NOTICE 'Survey Data Cleaner database initialized successfully';
END $$;
