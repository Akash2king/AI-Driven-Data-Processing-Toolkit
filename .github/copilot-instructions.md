# LLM-Guided Survey Data Cleaning Tool

This project is a comprehensive web-based interactive system that uses ML to generate metadata about survey datasets and employs an LLM to decide, verify, and control the execution of cleaning and processing steps.

## Project Structure

- **Backend**: Python FastAPI with modular architecture
- **Frontend**: React with TailwindCSS
- **ML Libraries**: pandas, scikit-learn, scipy, statsmodels
- **LLM Integration**: OpenAI GPT-4 API or Llama-3
- **Database**: PostgreSQL for step history and Redis for caching
- **File Storage**: Local storage with AWS S3 option

## Development Guidelines

- Follow modular architecture principles
- Implement comprehensive error handling
- Maintain LLM prompt templates in database
- Ensure stateless API design for scalability
- Include thorough logging and monitoring

## Completed Steps

- [x] Project structure creation
- [x] Backend API implementation
- [x] Frontend React components
- [x] ML processing modules
- [x] LLM integration
- [x] Database setup
- [x] Docker configuration
- [x] Testing and documentation
- [x] Setup scripts and deployment configuration

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd statathon

# Set up environment variables
cp backend/.env.example backend/.env
# Edit backend/.env with your configuration

# Start all services
docker-compose up -d
```

### Manual Setup

```bash
# Run the setup script
# On Windows:
setup.bat

# On macOS/Linux:
chmod +x setup.sh
./setup.sh
```

### Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Project Features Implemented

### Backend Modules ✅

- **data_upload.py**: File upload, validation, and preview functionality
- **metadata_generator.py**: Comprehensive dataset analysis and metadata generation
- **llm_controller.py**: LLM integration for decision making and verification
- **ml_processors.py**: ML-powered data cleaning operations
- **report_generator.py**: Comprehensive reporting and export functionality

### Frontend Components ✅

- **FileUpload**: Drag & drop file upload with validation
- **DataPreview**: Interactive data table with missing value highlighting
- **HomePage**: Main application interface with tabbed workflow

### Core Capabilities ✅

- Multi-format file support (CSV, Excel, SPSS)
- Automated data quality assessment
- LLM-guided cleaning recommendations
- Step-by-step user approval process
- Real-time verification and quality tracking
- Comprehensive audit trail and reporting
- Survey weight detection and analysis

### Infrastructure ✅

- PostgreSQL database with SQLAlchemy ORM
- Redis caching support
- Docker containerization
- Nginx reverse proxy configuration
- Comprehensive API documentation
- Health checks and monitoring

## Next Steps for Enhancement

- Implement advanced ML models for data imputation
- Add more sophisticated outlier detection algorithms
- Enhance the LLM prompt engineering for better recommendations
- Add user authentication and session management
- Implement advanced visualization components
- Add support for more file formats
- Enhance the reporting system with interactive charts
