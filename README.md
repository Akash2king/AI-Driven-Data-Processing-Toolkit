
# 📊 LLM-Guided Survey Data Cleaning and Preparation Tool

A **comprehensive web-based interactive system** that uses **machine learning** to generate metadata about survey datasets and employs a **Large Language Model (LLM)** to decide, verify, and control the execution of cleaning and processing steps — all with **step-by-step user acknowledgment** and **persistent memory** for quality assurance and reproducibility.

---

## 🚀 Features

### 🧠 Core Capabilities

* **Automated Data Analysis** – AI-powered detection of missing values, duplicates, outliers, and data quality issues.
* **LLM-Guided Decision Making** – GPT-4 or Llama-3 powered recommendations for cleaning steps.
* **Interactive Step Approval** – Users can approve, reject, or modify each cleaning action.
* **Comprehensive Reporting** – Before/after analysis with exportable cleaning reports.
* **Survey Weight Detection** – Automatic identification of survey weight variables.
* **Quality Scoring** – Real-time data quality assessment and improvement tracking.

### ⚙️ Technical Features

* **Scalable Architecture** – Stateless **FastAPI** backend with modular design.
* **Multiple File Formats** – Supports CSV, Excel (.xlsx, .xls), and SPSS (.sav).
* **Persistent Memory** – PostgreSQL storage for step history and LLM memory.
* **Real-time Verification** – LLM validates the effectiveness of each cleaning step.
* **Flexible Export Options** – Clean datasets exportable in multiple formats.

---

## 🏗️ System Architecture

### 🖥️ Backend (FastAPI + Python)

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
│   ├── core/             # Config & DB setup
│   ├── models/           # Database models
│   └── utils/            # Utility functions
├── main.py               # FastAPI entrypoint
└── requirements.txt      # Dependencies
```

### 🌐 Frontend (React + TypeScript)

```
frontend/
├── src/
│   ├── components/       # UI Components
│   │   ├── FileUpload.tsx
│   │   ├── DataPreview.tsx
│   │   ├── StepApprovalModal.tsx
│   │   ├── StepExecutionLog.tsx
│   │   └── FinalReportView.tsx
│   ├── pages/            # App Pages
│   ├── types/            # TS Types
│   ├── utils/            # API Helpers
│   └── App.tsx           # Main App Component
└── package.json          # Node.js dependencies
```

---

## 🛠️ Installation & Setup

### ✅ Prerequisites

* Python **3.8+**
* Node.js **16+**
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
cp .env.example .env  # Edit configuration
```

1. **Configure Database**

   * Create a PostgreSQL database.
   * Update `DATABASE_URL` in `.env`.
   * Run migrations (auto-created on startup).

2. **Start Backend**

```bash
python main.py
```

API will be available at **[http://localhost:8000](http://localhost:8000)**

---

### 🎨 Frontend Setup

```bash
cd frontend
npm install
npm start
```

Frontend will be available at **[http://localhost:3000](http://localhost:3000)**

---

## 🔧 Configuration

### `.env` Example

```env
DATABASE_URL=postgresql://user:password@localhost:5432/survey_cleaner
REDIS_URL=redis://localhost:6379

# LLM
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4
LLM_PROVIDER=openai  # or 'local'

UPLOAD_DIR=./data/uploads
MAX_FILE_SIZE=104857600  # 100MB
SECRET_KEY=your-super-secret-key
```

### LLM Options

* **OpenAI GPT-4 (Recommended):** `LLM_PROVIDER=openai`
* **Local Llama-3:** `LLM_PROVIDER=local` + configure `LOCAL_LLM_URL`

---

## 📊 Data Processing Pipeline

1. **Upload & Analyze**

   * File upload & validation
   * Metadata & quality score generation

2. **LLM Decision Making**

   * Recommendations for cleaning steps
   * Method selection based on dataset

3. **Interactive Cleaning**

   * Step-by-step user approval
   * Execution logs & live verification

4. **Quality Assurance**

   * Before/after comparison
   * Improvement tracking & lineage logs

5. **Reporting**

   * Generate and export final cleaning report

---

## 🎯 Supported Cleaning Operations

* **Duplicates:** Exact, near-match, fuzzy deduplication
* **Missing Values:** KNN, RF, statistical imputation
* **Outliers:** IQR, Z-score, Isolation Forest, LOF
* **Data Types:** Auto-detection & type conversion
* **Survey-Specific:** Weight detection, weighted stats, design effect analysis

---

## 🚀 Example API Workflow

```python
# Upload Dataset
POST /api/v1/cleaning/upload

# Create Session
POST /api/v1/cleaning/dataset/{dataset_id}/session

# Get LLM Recommendation
POST /api/v1/cleaning/session/{session_id}/next-step

# Execute Cleaning Step
POST /api/v1/cleaning/session/{session_id}/execute-step
```

Full documentation: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🧪 Testing

```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit & push changes
4. Submit a PR

---

## 📝 License

Licensed under **MIT License** – see [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

* **OpenAI** – GPT-4 API
* **Meta** – Llama-3 architecture
* **FastAPI & React** communities
* Survey research domain experts

---

## 📞 Support

* Open a GitHub issue
* Check `/docs`
* Review API at `/api/docs`

---

**Built with ❤️ for the survey research community**
