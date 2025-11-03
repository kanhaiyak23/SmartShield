# Dataset Setup Guide for SmartShield

## UNSW-NB15 Dataset

SmartShield uses the UNSW-NB15 dataset for training campus Wi-Fi anomaly detection models.

### Download Instructions

1. **Official Source**: UNSW-NB15 dataset can be obtained from:
   - University of New South Wales: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/
   - Alternative: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

2. **Required Files**:
   - `UNSW_NB15_training-set.csv` - Training data
   - `UNSW_NB15_testing-set.csv` - Testing data
   - Optional: `UNSW_NB15_features.csv` - Feature descriptions

3. **Place Files**:
   ```
   python/
   └── data/
       ├── UNSW_NB15_training-set.csv
       ├── UNSW_NB15_testing-set.csv
       └── UNSW_NB15_features.csv (optional)
   ```

### Dataset Information

**UNSW-NB15** is a comprehensive dataset with:
- **Normal traffic**: Legitimate network communication
- **Attack categories**:
  - Analysis (Port Scan, Vulnerability Scan)
  - Backdoor
  - DoS (Denial of Service)
  - Exploits
  - Fuzzers
  - Generic
  - Reconnaissance
  - Shellcode
  - Worms

### Dataset Statistics

- **Total Records**: ~2.5 million
- **Features**: 42 network features
- **Training set**: ~175,000 records
- **Testing set**: ~82,000 records

### Why UNSW-NB15?

1. **Campus-focused**: Designed for modern network environments
2. **Comprehensive**: Covers multiple attack types relevant to campus Wi-Fi
3. **Realistic**: Traffic patterns similar to campus networks
4. **Modern**: Represents contemporary attack techniques

### Using Alternative Datasets

To use CICIDS2017 or other datasets:

1. **Create a new preprocessor** in `src/preprocessing.py`:
   ```python
   def preprocess_cicids2017(self, df):
       # CICIDS2017-specific preprocessing
       pass
   ```

2. **Update training script** in `train_models.py`:
   ```python
   def prepare_cicids2017_dataset(data_dir='data'):
       # Load CICIDS2017 files
       pass
   ```

3. **Modify feature extraction** in `src/packet_capture.py` to match dataset features

### Quick Start

After downloading the dataset:

```bash
cd python
python train_models.py --train-all
```

This will:
1. Load UNSW-NB15 dataset
2. Preprocess features
3. Train Random Forest, Isolation Forest, and Autoencoder models
4. Save models to `models/` directory

### Verification

To verify dataset loading:

```python
import pandas as pd
df = pd.read_csv('data/UNSW_NB15_training-set.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Labels: {df['label'].value_counts()}")
print(f"Attacks: {df['attack_cat'].value_counts()}")
```

### Troubleshooting

**Issue**: "File not found: UNSW_NB15_training-set.csv"
- **Solution**: Ensure CSV files are in `python/data/` directory

**Issue**: "Memory error during training"
- **Solution**: Reduce dataset size or increase system RAM

**Issue**: "Feature mismatch"
- **Solution**: Check that feature extraction in `packet_capture.py` matches training features

### License

UNSW-NB15 is freely available for research purposes. Check the official UNSW website for usage terms.


