# LLM-Guided Survey Data Cleaning and Preparation Tool

A comprehensive web-based interactive system that uses machine learning to generate metadata about survey datasets and employs a Large Language Model (LLM) to decide, verify, and control the execution of cleaning and processing steps with step-by-step user acknowledgment and persistent memory for quality assurance and reproducibility.

## 🚀 Features

### Core Capabilities

- **Automated Data Analysis**: AI-powered analysis of datasets to identify missing values, duplicates, outliers, and data quality issues
- **LLM-Guided Decision Making**: GPT-4 or Llama-3 powered recommendations for cleaning steps and methods
- **Interactive Step Approval**: User-controlled execution with full transparency and verification
- **Comprehensive Reporting**: Detailed before/after analysis with exportable cleaning reports
- **Survey Weight Detection**: Automatic identification of potential survey weight variables
- **Quality Scoring**: Real-time data quality assessment and improvement tracking

### Technical Features

- **Scalable Architecture**: Stateless FastAPI backend with modular design
- **Multiple File Formats**: Support for CSV, Excel (.xlsx, .xls), and SPSS (.sav) files
- **Persistent Memory**: PostgreSQL storage for step history and LLM memory
- **Real-time Verification**: LLM verification of each cleaning step's effectiveness
- **Export Options**: Clean datasets exportable in multiple formats

## 🏗️ Architecture

### Backend (Python FastAPI)

```
backend/
├── app/
│   ├── modules/          # Core processing modules
│   │   ├── data_upload.py
│   │   ├── metadata_generator.py
│   │   ├── llm_controller.py
│   │   ├── ml_processors.py
│   │   └── report_generator.py
│   ├── api/              # API endpoints
│   ├── core/             # Configuration and database
│   ├── models/           # Database models
│   └── utils/            # Utility functions
├── main.py               # FastAPI application
└── requirements.txt      # Python dependencies
```

### Frontend (React + TypeScript)

```
frontend/
├── src/
│   ├── components/       # Reusable UI components
│   │   ├── FileUpload.tsx
│   │   ├── DataPreview.tsx
│   │   ├── StepApprovalModal.tsx
│   │   ├── StepExecutionLog.tsx
│   │   └── FinalReportView.tsx
│   ├── pages/            # Main application pages
│   ├── types/            # TypeScript type definitions
│   ├── utils/            # API calls and helpers
│   └── App.tsx           # Main application component
└── package.json          # Node.js dependencies
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- Node.js 16+
- PostgreSQL 12+
- Redis 6+ (optional, for caching)

### Backend Setup

1. **Clone and navigate to backend directory**

```bash
cd backend
```

2. **Create and activate virtual environment**

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Configure database**

- Create PostgreSQL database
- Update DATABASE_URL in .env
- Run migrations (auto-created on startup)

6. **Start the backend server**

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**

```bash
cd frontend
```

2. **Install dependencies**

```bash
npm install
```

3. **Start the development server**

```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## 🔧 Configuration

### Environment Variables (.env)

```env
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/survey_cleaner
REDIS_URL=redis://localhost:6379

# LLM Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4
LLM_PROVIDER=openai  # openai or local

# File Storage
UPLOAD_DIR=./data/uploads
MAX_FILE_SIZE=104857600  # 100MB

# Security
SECRET_KEY=your-super-secret-key-change-this-in-production
```

### LLM Provider Options

**OpenAI GPT-4 (Recommended)**

- Set `LLM_PROVIDER=openai`
- Provide `OPENAI_API_KEY`
- Choose model with `OPENAI_MODEL`

**Local LLM (Llama-3)**

- Set `LLM_PROVIDER=local`
- Configure `LOCAL_LLM_URL`
- Ensure local LLM server is running

## 📊 Data Processing Pipeline

### 1. Data Upload & Analysis

- File upload with format validation
- Automatic metadata generation
- Initial data quality assessment

### 2. LLM Decision Making

- Context-aware analysis of data issues
- Prioritized cleaning step recommendations
- Method selection based on data characteristics

### 3. Interactive Processing

- Step-by-step user approval process
- Real-time execution with progress tracking
- LLM verification of results

### 4. Quality Assurance

- Before/after comparison analysis
- Quality score tracking
- Comprehensive audit trail

### 5. Report Generation

- Detailed cleaning summary
- Data lineage documentation
- Exportable clean datasets

## 🎯 Supported Cleaning Operations

### Data Quality Issues

- **Duplicate Records**: Exact, near-match, and fuzzy duplicate removal
- **Missing Values**: KNN, Random Forest, statistical imputation methods
- **Outlier Detection**: IQR, Z-score, Isolation Forest, LOF methods
- **Data Type Issues**: Automatic type detection and conversion

### Survey-Specific Features

- **Weight Variable Detection**: Automatic identification of survey weights
- **Weighted Statistics**: Calculation of weighted means and proportions
- **Design Effect Analysis**: Survey design effect computation

## 🚀 Usage Example

### 1. Upload Dataset

```python
# Backend API
POST /api/v1/cleaning/upload
Content-Type: multipart/form-data
```

### 2. Create Processing Session

```python
POST /api/v1/cleaning/dataset/{dataset_id}/session
{
  "requirements": {
    "preserve_sample_size": true,
    "handle_missing_values": true,
    "remove_duplicates": true,
    "detect_outliers": true
  }
}
```

### 3. Get LLM Recommendations

```python
POST /api/v1/cleaning/session/{session_id}/next-step
```

### 4. Execute Approved Steps

```python
POST /api/v1/cleaning/session/{session_id}/execute-step
{
  "task": "remove_duplicates",
  "method": "exact_match",
  "columns": [],
  "user_approved": true
}
```

## 🔍 API Documentation

Full API documentation is available at `http://localhost:8000/docs` when the backend is running.

### Key Endpoints

- `POST /api/v1/cleaning/upload` - Upload dataset
- `GET /api/v1/cleaning/dataset/{id}/preview` - Get data preview
- `POST /api/v1/cleaning/dataset/{id}/session` - Create processing session
- `POST /api/v1/cleaning/session/{id}/next-step` - Get LLM recommendations
- `POST /api/v1/cleaning/session/{id}/execute-step` - Execute cleaning step
- `GET /api/v1/cleaning/session/{id}/history` - Get processing history
- `POST /api/v1/cleaning/session/{id}/generate-report` - Generate final report

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest
```

### Frontend Tests

```bash
cd frontend
npm test
```

## 🚀 Deployment

### Docker Deployment (Recommended)

1. **Build and run with Docker Compose**

```bash
docker-compose up -d
```

### Manual Deployment

1. **Backend Production Setup**

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

2. **Frontend Production Build**

```bash
npm run build
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- Meta for Llama-3 architecture
- FastAPI and React communities
- Survey research community for domain expertise

## 📞 Support

For questions, issues, or feature requests:

- Create an issue on GitHub
- Check the documentation at `/docs`
- Review the API documentation at `/api/docs`

---

**Built with ❤️ for the survey research community**
#   A I - D r i v e n - D a t a - P r o c e s s i n g - T o o l k i t  
 