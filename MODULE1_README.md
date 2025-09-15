# Module 1: Automated Data Cleaning & Processing

## Overview

This is **Module 1** of an automated data processing pipeline that uses open-source Large Language Models to intelligently clean and process survey datasets.

## Key Features

- ✅ **Automated Data Cleaning**: Uses ML algorithms for outlier detection, missing value imputation, and duplicate removal
- ✅ **Open-Source LLM Integration**: Powered by Meta Llama 3.1 8B (free tier) via OpenRouter
- ✅ **Government-Style Reporting**: Professional PDF reports with charts and metrics
- ✅ **Multiple Export Formats**: CSV, Excel, JSON, and PDF export capabilities
- ✅ **SQLite Database**: No complex database setup required
- ✅ **JSON Serialization Safety**: Handles numpy data types and floating-point edge cases

## Recent Fixes (Latest Update)

### 🔧 Critical Issues Resolved:

1. **Report Generation Error Fix**:

   - Fixed `"POST /api/v1/cleaning/session/.../generate-report HTTP/1.1" 500 Internal Server Error`
   - Added comprehensive JSON serialization handling for numpy types and infinity values
   - All report endpoints now work properly

2. **Database Configuration**:

   - Switched from PostgreSQL to SQLite for better compatibility
   - No database server setup required

3. **Open-Source LLM Integration**:

   - **Model Changed**: Now uses `meta-llama/llama-3.1-8b-instruct:free` (free tier)
   - **Provider**: OpenRouter (no direct OpenAI API key needed)
   - **Cost**: Completely free for the base model

4. **Module 1 Branding**:
   - Updated all titles to reflect "Module 1: Automated Data Cleaning & Processing"
   - Clear positioning as the first module in a larger pipeline

## Quick Start

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup

```bash
cd streamlit_frontend
pip install -r requirements.txt
streamlit run app.py --server.port 3001
```

### 3. Access the Application

- **Frontend**: http://localhost:3001
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Configuration

### LLM Configuration (.env file)

```bash
# Open-Source Model Configuration
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENAI_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

### Supported Models via OpenRouter:

- `meta-llama/llama-3.1-8b-instruct:free` (Free tier)
- `meta-llama/llama-3.1-70b-instruct` (Paid)
- `mistralai/mistral-7b-instruct:free` (Free tier)
- `anthropic/claude-3.5-sonnet` (Paid)

## System Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Streamlit UI      │    │    FastAPI Backend  │    │   SQLite Database   │
│   (Port 3001)      │────│    (Port 8000)      │────│   (survey_cleaner)  │
│                     │    │                     │    │                     │
│ - Data Upload       │    │ - ML Processing     │    │ - Session Storage   │
│ - Progress Tracking │    │ - LLM Integration   │    │ - Processing History│
│ - Report Generation │    │ - JSON Serialization│    │ - Metadata Storage  │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                      │
                           ┌─────────────────────┐
                           │   OpenRouter API    │
                           │                     │
                           │ - Llama 3.1 8B      │
                           │ - Free Tier Model   │
                           │ - No API Cost       │
                           └─────────────────────┘
```

## What's Working Now

- ✅ File upload and validation
- ✅ Metadata generation and analysis
- ✅ Automated step recommendation
- ✅ Step execution with ML processors
- ✅ **Report generation (FIXED)**
- ✅ PDF export functionality
- ✅ Government-style reporting
- ✅ Open-source LLM integration
- ✅ SQLite database operations

## Next Steps (Future Modules)

- **Module 2**: Advanced Analytics & Visualization
- **Module 3**: Statistical Analysis & Modeling
- **Module 4**: Automated Insights & Recommendations
- **Module 5**: Export & Integration Pipeline

## Technology Stack

- **Backend**: FastAPI, SQLAlchemy, Pandas, Scikit-learn
- **Frontend**: Streamlit, Plotly, Pandas
- **LLM**: Meta Llama 3.1 8B via OpenRouter
- **Database**: SQLite
- **ML Libraries**: NumPy, SciPy, Statsmodels

## API Endpoints

- `POST /api/v1/cleaning/upload` - Upload dataset
- `GET /api/v1/cleaning/dataset/{id}/preview` - Preview data
- `POST /api/v1/cleaning/dataset/{id}/session` - Create processing session
- `POST /api/v1/cleaning/session/{id}/next-step` - Get next recommended step
- `POST /api/v1/cleaning/session/{id}/execute-step` - Execute cleaning step
- `POST /api/v1/cleaning/session/{id}/generate-report` - Generate final report ✅ FIXED
- `GET /api/v1/cleaning/session/{id}/download` - Download processed data

## License

MIT License - Feel free to use and modify for your projects.
