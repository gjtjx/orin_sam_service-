#!/bin/bash
# Script to run the SAM service

set -e

echo "Starting Orin SAM Service..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

# Run the service
echo "Starting service on http://localhost:8000"
echo "API documentation: http://localhost:8000/docs"
python main.py
