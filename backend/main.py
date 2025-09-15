from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path

from app.core.config import settings
from app.core.database import engine, Base
from app.api import data_cleaning

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="LLM-Guided Survey Data Cleaning and Preparation Tool"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(
    data_cleaning.router,
    prefix=f"{settings.API_V1_STR}/cleaning",
    tags=["data-cleaning"]
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.VERSION}

# Serve uploaded files and reports
@app.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    """Serve uploaded files and generated reports"""
    full_path = Path(settings.UPLOAD_DIR) / file_path
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(full_path)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "LLM-Guided Survey Data Cleaning API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
