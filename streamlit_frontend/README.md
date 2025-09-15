# LLM-Guided Survey Data Cleaning Tool - Streamlit Frontend

A powerful, interactive web application built with Streamlit for cleaning and processing survey datasets using AI guidance.

## Features

### 🚀 **4-Step Workflow**

1. **📤 Upload Data** - Drag & drop CSV, Excel, or SPSS files
2. **👀 Preview & Analyze** - Interactive data exploration with quality metrics
3. **🧹 Clean & Process** - LLM-guided step-by-step cleaning with user approval
4. **📊 Results & Export** - Download cleaned data with comprehensive reports

### 🤖 **AI-Powered Intelligence**

- **Smart Recommendations**: LLM analyzes your data and suggests optimal cleaning steps
- **Interactive Approval**: Review and approve each cleaning operation
- **Quality Verification**: AI verifies each step's success and impact
- **Natural Language Explanations**: Understand why each step is recommended

### 📊 **Rich Visualizations**

- Missing data heatmaps and distribution charts
- Data type analysis and quality scores
- Before/after comparison metrics
- Interactive data preview tables

### 🔧 **Comprehensive Processing**

- **Missing Value Handling**: Intelligent imputation strategies
- **Duplicate Detection**: Advanced duplicate identification and removal
- **Outlier Detection**: Statistical and ML-based outlier identification
- **Data Type Optimization**: Automatic type inference and conversion

## Quick Start

### Option 1: Using Batch File (Windows)

```bash
# Navigate to the frontend directory
cd d:\statathon\streamlit_frontend

# Run the startup script
start_frontend.bat
```

### Option 2: Using PowerShell

```powershell
# Navigate to the frontend directory
cd d:\statathon\streamlit_frontend

# Run the PowerShell script
.\start_frontend.ps1
```

### Option 3: Manual Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the application
streamlit run app.py
```

## Access Points

- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000 (must be running)
- **API Documentation**: http://localhost:8000/docs

## System Requirements

### Backend Prerequisites

The FastAPI backend must be running for full functionality:

```bash
# Navigate to backend directory
cd d:\statathon\backend

# Activate virtual environment (if using)
.\venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r requirements.txt

# Start backend server
python main.py
```

### Frontend Dependencies

- Python 3.8+
- streamlit==1.28.1
- pandas==2.1.3
- plotly==5.17.0
- requests==2.31.0

## File Support

| Format | Extension   | Description            |
| ------ | ----------- | ---------------------- |
| CSV    | .csv        | Comma-separated values |
| Excel  | .xlsx, .xls | Microsoft Excel files  |
| SPSS   | .sav        | SPSS data files        |

**Maximum file size**: 100MB

## Architecture

```
┌─────────────────┐    HTTP/REST API    ┌──────────────────┐
│   Streamlit     │ ←─────────────────→ │   FastAPI        │
│   Frontend      │                     │   Backend        │
│   (Port 8501)   │                     │   (Port 8000)    │
└─────────────────┘                     └──────────────────┘
         │                                        │
         │                                        │
         ▼                                        ▼
┌─────────────────┐                     ┌──────────────────┐
│   User          │                     │   SQLite         │
│   Interaction   │                     │   Database       │
└─────────────────┘                     └──────────────────┘
```

## Key Features Detail

### 📊 **Interactive Data Analysis**

- Real-time data quality metrics
- Missing value patterns visualization
- Column type distribution analysis
- Statistical summaries and outlier detection

### 🤖 **LLM Integration**

- OpenAI GPT-4 / Claude 3.5 Sonnet integration
- Context-aware cleaning recommendations
- Natural language explanations for each step
- User approval workflow with reasoning

### 🔄 **Processing Pipeline**

1. **Data Upload & Validation**
2. **Metadata Generation & Analysis**
3. **LLM Decision Making**
4. **ML-Powered Processing**
5. **Quality Verification**
6. **User Approval & Feedback**
7. **Report Generation & Export**

### 📈 **Quality Tracking**

- Before/after data quality scores
- Step-by-step impact analysis
- Data preservation metrics
- Comprehensive audit trail

## Troubleshooting

### Common Issues

**1. Backend Connection Error**

- Ensure backend is running on http://localhost:8000
- Check if all backend dependencies are installed
- Verify no port conflicts

**2. File Upload Issues**

- Check file format is supported (CSV, Excel, SPSS)
- Verify file size is under 100MB
- Ensure file is not corrupted

**3. Processing Errors**

- Check backend logs for detailed error messages
- Verify OpenAI/OpenRouter API keys are configured
- Ensure sufficient system memory for large datasets

### Status Indicators

The sidebar shows real-time system status:

- ✅ **Backend Connected**: All systems operational
- ❌ **Backend Offline**: Backend server not reachable
- ❌ **Backend Error**: Backend responding with errors

## Advanced Configuration

### Streamlit Configuration

Edit `.streamlit/config.toml` to customize:

- Theme colors and fonts
- Server settings
- Performance optimization

### Backend Configuration

Configure backend settings in `backend/.env`:

- API keys for LLM services
- Database connections
- File storage paths

## Development

### File Structure

```
streamlit_frontend/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── start_frontend.bat     # Windows startup script
├── start_frontend.ps1     # PowerShell startup script
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── README.md             # This file
```

### Extending the Application

- Add new visualization components in the main app.py
- Create additional processing steps in the backend
- Enhance UI components with custom CSS/HTML
- Add new export formats and options

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review backend logs at `d:\statathon\backend\`
3. Ensure all dependencies are properly installed
4. Verify API configurations are correct

## Version Information

- **Frontend**: Streamlit 1.28.1
- **Backend**: FastAPI with Python 3.10+
- **AI Models**: OpenAI GPT-4 / Claude 3.5 Sonnet
- **Database**: SQLite with SQLAlchemy ORM
