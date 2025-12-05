# UNSW-NB15 Dataset Training Instructions

## Overview

The UNSW-NB15 dataset is a comprehensive network intrusion detection dataset created by the Australian Centre for Cyber Security (ACCS). It contains real network traffic with both normal and attack samples.

## Download Options

### Option 1: Kaggle (Recommended - Easiest)

1. **Create a Kaggle account** (if you don't have one):
   - Go to: https://www.kaggle.com/
   - Sign up for free

2. **Install Kaggle CLI:**
   ```bash
   pip install kaggle
   ```

3. **Get your API credentials:**
   - Go to: https://www.kaggle.com/account
   - Scroll to "API" section
   - Click "Create New API Token"
   - This downloads `kaggle.json`
   - Place it in `~/.kaggle/kaggle.json`:
     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

4. **Download the dataset:**
   ```bash
   mkdir -p data
   kaggle datasets download -d mrwellsdavid/unsw-nb15 -p data/
   cd data
   unzip unsw-nb15.zip
   ```

5. **Train the model:**
   ```bash
   cd ..
   source venv/bin/activate
   python3 train_unsw_nb15.py data/UNSW_NB15_training-set.csv
   ```

### Option 2: Direct Download from Official Source

1. **Visit the official UNSW-NB15 page:**
   - https://research.unsw.edu.au/projects/unsw-nb15-dataset

2. **Request access** (may require registration)

3. **Download the dataset files:**
   - Look for `UNSW_NB15_training-set.csv` or similar CSV files
   - The dataset is large (~500MB+)

4. **Place in data directory:**
   ```bash
   mkdir -p data
   # Move your downloaded file to:
   mv ~/Downloads/UNSW_NB15_training-set.csv data/
   ```

5. **Train the model:**
   ```bash
   source venv/bin/activate
   python3 train_unsw_nb15.py data/UNSW_NB15_training-set.csv
   ```

### Option 3: Alternative Sources

- **GitHub repositories:** Search for "UNSW-NB15" on GitHub
- **Academic repositories:** Check research paper supplementary materials
- **Mirror sites:** Various academic institutions host mirrors

## Dataset Information

- **Size:** ~500MB+ (compressed)
- **Format:** CSV files
- **Features:** 49 features including:
  - Network flow statistics
  - Protocol information
  - Port numbers
  - Packet sizes
  - Attack labels

## Training Process

The training script (`train_unsw_nb15.py`) will:

1. Load the CSV dataset
2. Extract the 5 features we use:
   - Packet length (from sbytes + dbytes)
   - TTL (defaulted to 64)
   - Protocol code (mapped from protocol name)
   - Source port
   - Destination port
3. Filter to normal traffic only (for Isolation Forest training)
4. Train the Isolation Forest model
5. Evaluate on attack samples
6. Save the model and scaler

## Expected Results

After training, you should see:
- Model trained on thousands of normal samples
- Attack detection rate (percentage of attacks flagged as anomalies)
- Model files saved: `isolation_forest_model.joblib` and `feature_scaler.joblib`

## Troubleshooting

**Issue:** "File not found"
- Make sure the CSV file is in the `data/` directory
- Check the filename matches what the script expects
- Use absolute path: `python3 train_unsw_nb15.py /full/path/to/file.csv`

**Issue:** "No features extracted"
- Check that the CSV has columns like: `sbytes`, `dbytes`, `proto`, `sport`, `dsport`
- The script tries to auto-detect column names, but datasets may vary

**Issue:** "Out of memory"
- The script limits to first 100,000 rows by default
- For full dataset, remove the `nrows=100000` limit in `train_unsw_nb15.py`

**Issue:** Dataset format differs
- UNSW-NB15 has multiple versions/formats
- You may need to adjust column names in `extract_features_from_unsw()` function

## Current Status

If you haven't downloaded the dataset yet, the system uses a **synthetic model** trained on simulated normal traffic patterns. This works for demonstration but a real dataset will provide better accuracy.

