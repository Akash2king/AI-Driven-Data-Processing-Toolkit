from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer)
    upload_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String, default="uploaded")  # uploaded, processing, completed, error
    dataset_metadata = Column(JSON)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    
    # Relationships
    processing_steps = relationship("ProcessingStep", back_populates="dataset")

class ProcessingStep(Base):
    __tablename__ = "processing_steps"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    step_number = Column(Integer, nullable=False)
    task = Column(String, nullable=False)  # remove_duplicates, impute_missing, etc.
    method = Column(String, nullable=False)
    columns = Column(JSON)  # List of columns affected
    parameters = Column(JSON)  # Method-specific parameters
    pre_metadata = Column(JSON)
    post_metadata = Column(JSON)
    execution_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    execution_time = Column(Integer)  # seconds
    status = Column(String, default="pending")  # pending, executing, completed, failed, verified
    llm_reasoning = Column(Text)
    verification_status = Column(String)  # confirmed, needs_review, failed
    verification_reason = Column(Text)
    user_approved = Column(Boolean, default=False)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="processing_steps")

class LLMPromptTemplate(Base):
    __tablename__ = "llm_prompt_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    version = Column(String, nullable=False)
    template = Column(Text, nullable=False)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_timestamp = Column(DateTime(timezone=True), server_default=func.now())

class ProcessingSession(Base):
    __tablename__ = "processing_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    session_id = Column(String, unique=True, nullable=False)
    requirements = Column(JSON)  # User-specified cleaning requirements
    current_step = Column(Integer, default=0)
    status = Column(String, default="active")  # active, completed, paused, cancelled
    created_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    completed_timestamp = Column(DateTime(timezone=True))
