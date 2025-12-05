#!/bin/bash
# Helper script to download UNSW-NB15 dataset

echo "UNSW-NB15 Dataset Download Helper"
echo "=================================="
echo ""

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "Option 1: Download from Kaggle (Recommended)"
echo "---------------------------------------------"
echo "1. Install Kaggle CLI: pip install kaggle"
echo "2. Get your API token from: https://www.kaggle.com/account"
echo "3. Place kaggle.json in ~/.kaggle/"
echo "4. Run: kaggle datasets download -d mrwellsdavid/unsw-nb15 -p $DATA_DIR"
echo "5. Unzip: unzip $DATA_DIR/unsw-nb15.zip -d $DATA_DIR/"
echo ""

echo "Option 2: Direct Download"
echo "-------------------------"
echo "Download from: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15"
echo "Extract and place UNSW_NB15_training-set.csv in $DATA_DIR/"
echo ""

echo "Option 3: Using wget/curl"
echo "-------------------------"
echo "Try downloading from official source:"
echo "wget https://www.unsw.adfa.edu.au/unsw-cs-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/NUSW-NB15_features.csv -O $DATA_DIR/UNSW_NB15_training-set.csv"
echo ""

# Try direct download
echo "Attempting automatic download..."
if command -v wget &> /dev/null; then
    echo "Using wget..."
    wget -O "$DATA_DIR/UNSW_NB15_training-set.csv" "https://www.unsw.adfa.edu.au/unsw-cs-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/NUSW-NB15_features.csv" 2>&1 | head -20
elif command -v curl &> /dev/null; then
    echo "Using curl..."
    curl -L -o "$DATA_DIR/UNSW_NB15_training-set.csv" "https://www.unsw.adfa.edu.au/unsw-cs-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/NUSW-NB15_features.csv" 2>&1 | head -20
else
    echo "wget/curl not found. Please download manually."
fi

if [ -f "$DATA_DIR/UNSW_NB15_training-set.csv" ]; then
    SIZE=$(du -h "$DATA_DIR/UNSW_NB15_training-set.csv" | cut -f1)
    echo ""
    echo "✅ File downloaded: $DATA_DIR/UNSW_NB15_training-set.csv ($SIZE)"
    echo "Now run: python3 train_unsw_nb15.py $DATA_DIR/UNSW_NB15_training-set.csv"
else
    echo ""
    echo "⚠️  Automatic download failed. Please download manually."
    echo "See instructions above."
fi

