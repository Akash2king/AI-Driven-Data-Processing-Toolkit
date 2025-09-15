import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Module 1: Automated Data Cleaning & Processing"
    VERSION: str = "1.0.0"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Database - Using SQLite for better compatibility
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./survey_cleaner.db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # File Storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: list = [".csv", ".xlsx", ".xls", ".sav"]
    
    # LLM Configuration - Updated to use open-source models via OpenRouter
    OPENAI_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")  # Using OpenRouter API key
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "meta-llama/llama-3.1-8b-instruct:free")  # Free open-source model
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter")  # openai, openrouter, or local
    LOCAL_LLM_URL: str = os.getenv("LOCAL_LLM_URL", "http://localhost:8000")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # AWS S3 (Optional)
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "")
    
    # CORS
    BACKEND_CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:3001"]
    
    class Config:
        env_file = ".env"

settings = Settings()
