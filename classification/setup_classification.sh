#!/bin/bash

# Quick Start Script for Hoop.io Player Classification
# This script sets up and trains the player classification model

set -e  # Exit on error

echo "=========================================="
echo "Hoop.io Player Classification Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "../backend/requirements.txt" ]; then
    echo "Error: Please run this script from the classification directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "Creating virtual environment..."
    cd ..
    python3 -m venv venv
    cd classification
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source ../venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r ../backend/requirements.txt --quiet
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p ../data/raw
mkdir -p ../data/models
echo "✓ Directories created"

# Step 1: Download and Cache NBA Data
echo ""
echo "=========================================="
echo "Step 1: Downloading NBA Data (5 Seasons)"
echo "=========================================="
echo ""
echo "This will download and cache player statistics from"
echo "2021-22 through 2025-26 seasons (if not already cached)..."
echo ""

python download_nba_data.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Data download/cache completed successfully!"
else
    echo ""
    echo "✗ Data download failed. Please check the errors above."
    exit 1
fi

# Step 3: Model Training
echo ""
echo "=========================================="
echo "Step 3: Training Neural Network"
echo "=========================================="
echo ""
echo "This will process cached data from all 5 seasons"
echo "and prepare it for training..."
echo ""
python data_preprocessing.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Data preprocessing completed successfully!"
else
    echo ""
    echo "✗ Data preprocessing failed. Please check the errors above."
    exit 1
fi

# Step 2: Model Training
echo ""
echo "=========================================="
echo "Step 2: Training Neural Network"
echo "=========================================="
echo ""
echo "This will train the player classification model."
echo "Training typically takes 2-5 minutes..."
echo ""

python train_classifier.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Model training completed successfully!"
else
    echo ""
    echo "✗ Model training failed. Please check the errors above."
    exit 1
fi

cd ..

# Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Your player classification model is ready to use!"
echo ""
echo "Files created:"
echo "  📁 data/raw/player_stats_*.csv (5 season files)"
echo "  📁 data/raw/player_stats_combined.csv"
echo "  📁 data/X_train.npy"
echo "  📁 data/y_train.npy"
echo "  📁 data/scaler_params.json"
echo "  📁 data/models/player_classifier.pth"
echo "  📁 data/models/training_history.png"
echo "  📁 data/models/confusion_matrix.png"
echo ""
echo "Model trained on 5 seasons (2021-26) for better performance!"
echo "Current season (2025-26, started Sept 21) will be used for classification."
echo ""
echo "Next steps:"
echo "  1. Start the backend: uvicorn backend.main:app --reload"
echo "  2. Start the frontend: cd frontend && npm run dev"
echo "  3. Try: 'Classify Stephen Curry' or 'Tonight's game: who are the Elite tier players?'"
echo ""
echo "Example questions for current games:"
echo "  - 'Which team has more Elite players in tonight's game?'"
echo "  - 'Classify the starters for both teams playing now'"
echo "  - 'Show tier breakdown for today's matchup'"
echo ""
echo "For more information, see CLASSIFICATION_GUIDE.md"
echo ""
echo "=========================================="
