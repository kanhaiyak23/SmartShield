#!/bin/bash
# Helper script to set up Kaggle credentials

echo "Setting up Kaggle for UNSW-NB15 Dataset Download"
echo "================================================"
echo ""

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "Installing Kaggle CLI..."
    pip install kaggle
fi

echo ""
echo "📋 Instructions to get Kaggle credentials:"
echo "1. Go to: https://www.kaggle.com/account"
echo "2. Scroll down to 'API' section"
echo "3. Click 'Create New API Token'"
echo "4. This will download 'kaggle.json'"
echo ""
echo "Once you have kaggle.json:"
echo "1. Run: mkdir -p ~/.kaggle"
echo "2. Run: mv ~/Downloads/kaggle.json ~/.kaggle/"
echo "3. Run: chmod 600 ~/.kaggle/kaggle.json"
echo ""
echo "Then download the dataset:"
echo "  kaggle datasets download -d mrwellsdavid/unsw-nb15 -p data/"
echo "  cd data && unzip unsw-nb15.zip && cd .."
echo ""
echo "Or if you already have the CSV files, place them in the data/ directory:"
echo "  - data/UNSW-NB15_1.csv (or)"
echo "  - data/UNSW-NB15_2.csv (or)"
echo "  - data/UNSW-NB15_3.csv (or)"
echo "  - data/UNSW-NB15_4.csv (or)"
echo "  - data/UNSW_NB15_training-set.csv"
echo ""

