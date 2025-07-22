#!/usr/bin/env python
"""
Script for making predictions with trained fraud detection models.
"""
import os
import argparse
import joblib
import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('prediction.log')  # Output to file
    ]
)
logger = logging.getLogger(__name__)

# Define directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Make predictions with trained fraud detection models")
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the trained model file'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to the data file to make predictions on'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=os.path.join(RESULTS_DIR, f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'),
        help='Path to save the predictions'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Probability threshold for classification'
    )
    
    return parser.parse_args()


def load_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        model: Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    return model


def load_data(data_path):
    """
    Load data for prediction.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        X: Features for prediction
    """
    logger.info(f"Loading data from {data_path}")
    
    # Determine file type and load accordingly
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.npy'):
        data = np.load(data_path, allow_pickle=True)
    else:
        logger.error(f"Unsupported file format: {data_path}")
        raise ValueError(f"Unsupported file format: {data_path}")
    
    logger.info(f"Loaded data with shape {data.shape}")
    return data


def save_predictions(predictions, probabilities, output_path):
    """
    Save predictions to disk.
    
    Args:
        predictions: Predicted labels
        probabilities: Prediction probabilities
        output_path: Path to save the predictions
        
    Returns:
        str: Path to saved predictions
    """
    logger.info(f"Saving predictions to {output_path}")
    
    # Create DataFrame with predictions
    df = pd.DataFrame({
        'predicted_class': predictions,
        'probability': probabilities
    })
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=True)
    
    return output_path


def make_predictions(model, X, threshold=0.5):
    """
    Make predictions with a trained model.
    
    Args:
        model: Trained model
        X: Features for prediction
        threshold: Probability threshold for classification
        
    Returns:
        tuple: (predictions, probabilities)
    """
    logger.info("Making predictions...")
    
    # Get probability predictions
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
    else:
        predictions = model.predict(X)
        probabilities = predictions.astype(float)
    
    # Count fraud predictions
    n_fraud = np.sum(predictions == 1)
    fraud_rate = n_fraud / len(predictions) * 100
    
    logger.info(f"Made {len(predictions)} predictions")
    logger.info(f"Predicted {n_fraud} fraudulent transactions ({fraud_rate:.2f}%)")
    
    return predictions, probabilities


def run_prediction(args):
    """
    Main prediction function.
    
    Args:
        args: Command line arguments
    """
    # Load model
    model = load_model(args.model)
    
    # Load data
    X = load_data(args.data)
    
    # Make predictions
    predictions, probabilities = make_predictions(model, X, args.threshold)
    
    # Save predictions
    output_path = save_predictions(predictions, probabilities, args.output)
    logger.info(f"Predictions saved to {output_path}")
    
    return predictions, probabilities


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Fraud Detection Prediction")
    logger.info("="*60)
    
    # Parse command line arguments
    args = parse_args()
    logger.info(f"Arguments: {args}")
    
    # Run prediction
    predictions, probabilities = run_prediction(args) 