@echo off
echo Fixing Streamlit compatibility issues...

echo Upgrading Streamlit to latest version...
cd streamlit_frontend
pip install --upgrade streamlit>=1.32.0

echo Streamlit updated successfully!
echo You can now run the application with: streamlit run app.py

pause
