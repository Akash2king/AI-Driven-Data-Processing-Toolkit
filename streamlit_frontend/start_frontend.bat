@echo off
echo Starting LLM-Guided Survey Data Cleaning Tool - Streamlit Frontend
echo.

echo Installing/Updating Python packages...
pip install -r requirements.txt

echo.
echo Starting Streamlit application...
echo Frontend will be available at: http://localhost:8501
echo Backend should be running at: http://localhost:8000
echo.

streamlit run app.py
