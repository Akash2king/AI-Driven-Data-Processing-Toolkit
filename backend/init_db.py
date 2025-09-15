"""
Database initialization script for LLM Survey Data Cleaning Tool
"""
import os
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from sqlalchemy import create_engine, text
from app.core.config import settings
from app.core.database import Base
from app.models.models import Dataset, ProcessingStep, LLMPromptTemplate, ProcessingSession

def create_database():
    """Create database and tables"""
    try:
        # Create engine
        engine = create_engine(settings.DATABASE_URL)
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        print("✅ Database tables created successfully")
        
        # Insert default LLM prompt templates
        insert_default_prompts(engine)
        
        return True
        
    except Exception as e:
        print(f"❌ Database creation failed: {e}")
        return False

def insert_default_prompts(engine):
    """Insert default LLM prompt templates"""
    from sqlalchemy.orm import sessionmaker
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Check if templates already exist
        existing = db.query(LLMPromptTemplate).filter(LLMPromptTemplate.name == "decide_step").first()
        if existing:
            print("📝 LLM prompt templates already exist")
            return
        
        # Default prompt templates
        templates = [
            {
                "name": "decide_step",
                "version": "1.0",
                "description": "Prompt for deciding next cleaning step",
                "template": """
You are the decision brain for a survey dataset cleaning tool. Given the following information, decide which cleaning steps to execute next.

Dataset Metadata:
{metadata}

User Requirements:
{requirements}

Processing History:
{history}

Available Methods:
{available_methods}

Based on this information, decide the next cleaning step(s). Consider:
1. Data quality issues that need addressing
2. User requirements and priorities
3. Logical sequence of operations
4. Risk of data loss or corruption

Respond with a JSON object in this exact format:
{{
  "steps": [
    {{
      "task": "task_name",
      "method": "method_name",
      "columns": ["column1", "column2"],
      "parameters": {{"param1": "value1"}},
      "priority": "high|medium|low",
      "risk_level": "low|medium|high"
    }}
  ],
  "reasoning": "Explanation of why these steps were chosen",
  "confidence": 0.8
}}
"""
            },
            {
                "name": "verify_step",
                "version": "1.0",
                "description": "Prompt for verifying step execution",
                "template": """
You are the quality control agent for a data cleaning operation. Compare the before and after states to verify the operation was successful.

Pre-Step Metadata:
{pre_metadata}

Post-Step Metadata:
{post_metadata}

Step Information:
{step_info}

Processing History:
{history}

Analyze the changes and determine:
1. Did the intended change occur correctly?
2. Were there any unexpected side effects?
3. Is the data quality better or worse?
4. Should this step be approved?

Respond with a JSON object in this exact format:
{{
  "status": "confirmed|needs_review|failed",
  "reason": "Detailed explanation of the verification result",
  "confidence": 0.9,
  "side_effects": ["list", "of", "unexpected", "changes"],
  "quality_impact": "positive|neutral|negative"
}}
"""
            }
        ]
        
        # Insert templates
        for template_data in templates:
            template = LLMPromptTemplate(**template_data)
            db.add(template)
        
        db.commit()
        print("✅ Default LLM prompt templates inserted")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Failed to insert prompt templates: {e}")
    finally:
        db.close()

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import psycopg2
        print("✅ PostgreSQL driver (psycopg2) is available")
    except ImportError:
        print("❌ PostgreSQL driver (psycopg2) not found. Install with: pip install psycopg2-binary")
        return False
    
    try:
        import redis
        print("✅ Redis driver is available")
    except ImportError:
        print("⚠️ Redis driver not found. Install with: pip install redis")
        print("   Redis is optional but recommended for caching")
    
    return True

def test_connections():
    """Test database and Redis connections"""
    # Test PostgreSQL connection
    try:
        engine = create_engine(settings.DATABASE_URL)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ PostgreSQL connection successful")
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False
    
    # Test Redis connection (optional)
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        print("✅ Redis connection successful")
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}")
        print("   Redis is optional but recommended for caching")
    
    return True

if __name__ == "__main__":
    print("🚀 Initializing LLM Survey Data Cleaning Tool Database")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Test connections
    if not test_connections():
        print("\n❌ Connection tests failed. Please check your configuration.")
        sys.exit(1)
    
    # Create database
    if create_database():
        print("\n✅ Database initialization completed successfully!")
        print(f"   Database URL: {settings.DATABASE_URL}")
        print(f"   Redis URL: {settings.REDIS_URL}")
        print("\n🎉 You can now start the application with: python main.py")
    else:
        print("\n❌ Database initialization failed!")
        sys.exit(1)
