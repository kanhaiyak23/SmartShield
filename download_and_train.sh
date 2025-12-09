#!/bin/bash
# Automated script to download UNSW-NB15 dataset and train Random Forest model

set -e

echo "=========================================="
echo "SmartShield - Dataset Download & Training"
echo "=========================================="
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Check for Kaggle credentials
KAGGLE_CREDENTIALS="$HOME/.kaggle/kaggle.json"
if [ ! -f "$KAGGLE_CREDENTIALS" ]; then
    echo ""
    echo "❌ Kaggle credentials not found!"
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to: https://www.kaggle.com/account"
    echo "2. Scroll to 'API' section"
    echo "3. Click 'Create New API Token' (downloads kaggle.json)"
    echo "4. Run these commands:"
    echo "   mkdir -p ~/.kaggle"
    echo "   mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "   chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ Kaggle credentials found"

# Install Kaggle CLI if not present
if ! command -v kaggle &> /dev/null; then
    echo "Installing Kaggle CLI..."
    pip install kaggle -q
fi

# Create data directory
mkdir -p data

# Download dataset
echo ""
echo "Downloading UNSW-NB15 dataset from Kaggle..."
echo "This may take a few minutes (dataset is ~500MB)..."
kaggle datasets download -d mrwellsdavid/unsw-nb15 -p data/

# Extract
echo ""
echo "Extracting dataset..."
cd data
if [ -f "unsw-nb15.zip" ]; then
    unzip -q unsw-nb15.zip
    echo "✅ Dataset extracted"
else
    echo "❌ Download failed. Please check your Kaggle credentials and internet connection."
    exit 1
fi
cd ..

# Check what files we have
echo ""
echo "Checking downloaded files..."
ls -lh data/*.csv 2>/dev/null | head -10 || echo "No CSV files found in expected format"

# Find training files
TRAINING_FILES=""
if [ -f "data/UNSW_NB15_training-set.csv" ]; then
    TRAINING_FILES="data/UNSW_NB15_training-set.csv"
elif [ -f "data/UNSW-NB15_1.csv" ]; then
    TRAINING_FILES="data/UNSW-NB15_1.csv data/UNSW-NB15_2.csv data/UNSW-NB15_3.csv data/UNSW-NB15_4.csv"
else
    # Try to find any CSV files
    TRAINING_FILES=$(ls data/*.csv 2>/dev/null | head -5 | tr '\n' ' ')
fi

if [ -z "$TRAINING_FILES" ]; then
    echo "❌ No training CSV files found in data/ directory"
    echo "Please check the extracted files and manually specify the training file path"
    exit 1
fi

echo "Found training files: $TRAINING_FILES"

# Train the model
echo ""
echo "=========================================="
echo "Starting Random Forest Training"
echo "=========================================="
echo "This will take approximately 45 minutes..."
echo ""

python3 train_enhanced_rf.py

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Model files created:"
ls -lh *.joblib 2>/dev/null || echo "No .joblib files found (training may have failed)"
echo ""
echo "You can now restart the server to use the trained model:"
echo "  sudo ./start-server.sh"

