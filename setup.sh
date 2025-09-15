#!/bin/bash

# LLM Survey Data Cleaning Tool - Development Setup Script
# This script sets up the development environment

set -e

echo "🚀 Setting up LLM Survey Data Cleaning Tool"
echo "============================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        python_version=$(python3 --version 2>&1 | awk '{print $2}')
        echo "✅ Python version: $python_version"
    else
        echo "❌ Python 3 is not installed"
        exit 1
    fi
}

# Function to check Node.js version
check_node_version() {
    if command_exists node; then
        node_version=$(node --version)
        echo "✅ Node.js version: $node_version"
    else
        echo "❌ Node.js is not installed"
        exit 1
    fi
}

# Check prerequisites
echo "🔍 Checking prerequisites..."
check_python_version
check_node_version

if command_exists docker; then
    echo "✅ Docker is available"
else
    echo "⚠️ Docker is not available - you can still run the app manually"
fi

if command_exists psql; then
    echo "✅ PostgreSQL client is available"
else
    echo "⚠️ PostgreSQL client not found - please install PostgreSQL"
fi

echo ""

# Setup backend
echo "🐍 Setting up Python backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️ Please edit .env file with your configuration"
fi

# Initialize database
echo "Initializing database..."
python init_db.py

cd ..

# Setup frontend
echo ""
echo "⚛️ Setting up React frontend..."
cd frontend

# Install dependencies
echo "Installing Node.js dependencies..."
npm install

cd ..

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 To start the application:"
echo "1. Start the backend:"
echo "   cd backend && source venv/bin/activate && python main.py"
echo ""
echo "2. Start the frontend (in another terminal):"
echo "   cd frontend && npm start"
echo ""
echo "3. Or use Docker:"
echo "   docker-compose up -d"
echo ""
echo "📱 The application will be available at:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo ""
echo "⚙️ Don't forget to:"
echo "   - Set your OpenAI API key in backend/.env"
echo "   - Configure your PostgreSQL database settings"
echo "   - Review and update configuration as needed"
