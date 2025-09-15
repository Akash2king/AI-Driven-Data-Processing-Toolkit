
# 📊 LLM-Guided Survey Data Cleaning and Preparation Tool

A **web-based interactive system** that uses **machine learning** to analyze survey datasets and employs a **Large Language Model (LLM)** to guide, verify, and control data cleaning — with **step-by-step user approval** and **persistent memory** for reproducibility and auditability.

---

## 🚀 Features

### 🧠 Core Capabilities

* **Automated Data Analysis** – Detect missing values, duplicates, outliers, and type inconsistencies.
* **LLM-Guided Cleaning** – GPT-4 or Llama-3 powered recommendations for best cleaning strategies.
* **Interactive Step Control** – Users approve each step before execution.
* **Quality Tracking** – Real-time data quality scoring and improvement history.
* **Survey-Aware Features** – Detect survey weights, compute weighted stats, design effect analysis.
* **Comprehensive Reporting** – Full before/after comparison, downloadable reports.

---

## 🏗️ Architecture

### 🖥️ Backend (FastAPI)

```
backend/
├── app/
│   ├── modules/
│   │   ├── data_upload.py
│   │   ├── metadata_generator.py
│   │   ├── llm_controller.py
│   │   ├── ml_processors.py
│   │   └── report_generator.py
│   ├── api/
│   ├── core/
│   ├── models/
│   └── utils/
├── main.py
└── requirements.txt
```

### 🎨 Frontend (Streamlit)

```
frontend/
├── app.py              # Main Streamlit application
├── components/         # Custom UI components
│   ├── file_upload.py
│   ├── data_preview.py
│   ├── step_approval.py
│   ├── execution_log.py
│   └── final_report.py
├── utils/
│   └── api_client.py   # Functions for calling FastAPI endpoints
└── requirements.txt
```

---

## 🛠️ Installation & Setup

### ✅ Prerequisites

* Python **3.8+**
* PostgreSQL **12+**
* Redis **6+** *(optional, for caching)*

---

### 🔧 Backend Setup

```bash
cd backend
python -m venv venv
# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env with DB and LLM settings
```

1. **Configure database**

   * Create a PostgreSQL database.
   * Set `DATABASE_URL` in `.env`.
   * Run migrations (auto-applied on startup).

2. **Start backend**

```bash
python main.py
```

API available at **[http://localhost:8000](http://localhost:8000)**

---

### 🎨 Frontend Setup (Streamlit)

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Frontend available at **[http://localhost:8501](http://localhost:8501)**

---

## 🔧 Configuration

### `.env` Example (Backend)

```env
DATABASE_URL=postgresql://user:password@localhost:5432/survey_cleaner
REDIS_URL=redis://localhost:6379

OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4
LLM_PROVIDER=openai  # or 'local'

UPLOAD_DIR=./data/uploads
MAX_FILE_SIZE=104857600  # 100MB
SECRET_KEY=your-super-secret-key
```

---

## 📊 Data Processing Pipeline

1. **Upload & Analyze**

   * Streamlit upload → FastAPI validation
   * Automatic metadata & quality assessment

2. **LLM Decision Making**

   * GPT-4/Llama-3 suggests prioritized cleaning steps
   * Displays explanation for user approval

3. **Interactive Cleaning**

   * Approve/reject steps in Streamlit
   * View logs and intermediate results

4. **Quality Assurance**

   * Compare before/after data
   * Track quality improvement score

5. **Generate Report**

   * Export clean dataset + full audit trail

---

## 🎯 Supported Cleaning Operations

* **Duplicate Records** – Exact, near-match, fuzzy deduplication
* **Missing Values** – KNN, Random Forest, mean/median imputation
* **Outliers** – IQR, Z-score, Isolation Forest, LOF
* **Data Type Fixes** – Automatic type detection & conversion
* **Survey Weights** – Detect and compute weighted statistics

---

## 🚀 Example Workflow (API)

```python
# Upload Dataset
POST /api/v1/cleaning/upload

# Create Processing Session
POST /api/v1/cleaning/dataset/{dataset_id}/session

# Get LLM Recommendations
POST /api/v1/cleaning/session/{session_id}/next-step

# Execute Approved Cleaning Step
POST /api/v1/cleaning/session/{session_id}/execute-step
```

Full API documentation: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🧪 Testing

```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
pytest  # if using streamlit testing framework or unit tests
```

---

## 🚀 Deployment

### Docker Deployment (Recommended)

```bash
docker-compose up -d
```

### Manual Deployment

* **Backend:**

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

* **Frontend:**

```bash
streamlit run app.py --server.port 8501
```

---

## 🤝 Contributing

1. Fork this repository
2. Create a feature branch
3. Add/modify code
4. Write tests for new functionality
5. Submit a Pull Request

---

## 📝 License

Licensed under the **MIT License** – see [LICENSE](LICENSE).
