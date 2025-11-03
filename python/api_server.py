"""Flask API server for ML model predictions."""
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import DataPreprocessor
from src.models import ModelTrainer
from src.utils import setup_logging

logger = setup_logging()
app = Flask(__name__)
CORS(app)

# Global variables
preprocessor = None
model_trainer = None

def initialize_models():
    """Initialize preprocessor and models."""
    global preprocessor, model_trainer
    
    logger.info("Initializing models...")
    
    # Load preprocessor
    import joblib
    preprocessor_path = os.path.join('models', 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Loaded preprocessor from {preprocessor_path}")
    else:
        logger.error(f"Preprocessor not found at {preprocessor_path}")
        preprocessor = DataPreprocessor()
    
    # Load models
    model_trainer = ModelTrainer(models_dir='models')
    model_trainer.load_models()
    logger.info("Models loaded successfully")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': model_trainer is not None and preprocessor is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict attack type and detect anomalies."""
    try:
        data = request.json
        packet_features = data.get('features', {})
        
        if not packet_features:
            return jsonify({'error': 'No features provided'}), 400
        
        # Preprocess features
        processed_features = preprocessor.preprocess_packet_features(packet_features)
        
        # Get predictions
        results = model_trainer.combine_predictions(processed_features.iloc[0])
        
        return jsonify({
            'success': True,
            'prediction': results
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict for multiple packets."""
    try:
        data = request.json
        packets = data.get('packets', [])
        
        if not packets:
            return jsonify({'error': 'No packets provided'}), 400
        
        results = []
        for packet_features in packets:
            try:
                processed_features = preprocessor.preprocess_packet_features(packet_features)
                prediction = model_trainer.combine_predictions(processed_features.iloc[0])
                results.append({
                    'success': True,
                    'prediction': prediction
                })
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information."""
    info = {
        'models': [],
        'preprocessor': preprocessor is not None
    }
    
    if model_trainer:
        if model_trainer.random_forest is not None:
            info['models'].append('Random Forest')
        if model_trainer.isolation_forest is not None:
            info['models'].append('Isolation Forest')
        if model_trainer.autoencoder is not None:
            info['models'].append('Autoencoder')
    
    return jsonify(info)

if __name__ == '__main__':
    initialize_models()
    logger.info("Starting Flask API server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)

