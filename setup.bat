@echo off
REM LLM Survey Data Cleaning Tool - Windows Development Setup Script

echo 🚀 Setting up LLM Survey Data Cleaning Tool
echo =============================================

REM Check Python
echo 🔍 Checking prerequisites...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
) else (
    echo ✅ Python is available
)

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed or not in PATH
    pause
    exit /b 1
) else (
    echo ✅ Node.js is available
)

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Docker is not available - you can still run the app manually
) else (
    echo ✅ Docker is available
)

echo.

REM Setup backend
echo 🐍 Setting up Python backend...
cd backend

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating .env file...
    copy .env.example .env
    echo ⚠️ Please edit .env file with your configuration
)

REM Initialize database
echo Initializing database...
python init_db.py

cd ..

REM Setup frontend
echo.
echo ⚛️ Setting up React frontend...
cd frontend

REM Install dependencies
echo Installing Node.js dependencies...
npm install

cd ..

echo.
echo ✅ Setup completed successfully!
echo.
echo 🚀 To start the application:
echo 1. Start the backend:
echo    cd backend ^&^& venv\Scripts\activate ^&^& python main.py
echo.
echo 2. Start the frontend (in another terminal):
echo    cd frontend ^&^& npm start
echo.
echo 3. Or use Docker:
echo    docker-compose up -d
echo.
echo 📱 The application will be available at:
echo    Frontend: http://localhost:3000
echo    Backend API: http://localhost:8000
echo    API Documentation: http://localhost:8000/docs
echo.
echo ⚙️ Don't forget to:
echo    - Set your OpenAI API key in backend\.env
echo    - Configure your PostgreSQL database settings
echo    - Review and update configuration as needed
echo.
pause
