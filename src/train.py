#!/usr/bin/env python
"""
Main training script for fraud detection models.
"""
import os
import argparse
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import project modules
from data.loader import load_raw_data, load_processed_data, prepare_data
from models.models import (
    LogisticRegressionModel, 
    RandomForestModel,
    XGBoostModel,
    LightGBMModel
)
from utils.evaluation import (
    evaluate_and_plot,
    save_evaluation_results
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('training.log')  # Output to file
    ]
)
logger = logging.getLogger(__name__)

# Define directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['ecommerce', 'creditcard', 'both'],
        default='both',
        help='Which dataset to use for training'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        choices=['logistic', 'rf', 'xgb', 'lgb', 'all'],
        default='all',
        help='Which models to train'
    )
    
    parser.add_argument(
        '--use-resampled',
        action='store_true',
        help='Use SMOTE resampled data for training'
    )
    
    parser.add_argument(
        '--prepare-data',
        action='store_true',
        help='Prepare data from raw files (if False, load preprocessed data)'
    )
    
    return parser.parse_args()


def prepare_datasets(args):
    """
    Load and prepare datasets based on command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        dict: Dictionary of datasets
    """
    if args.prepare_data:
        # Prepare data from raw files
        logger.info("Loading raw data and preparing datasets...")
        fraud_df, cc_df = load_raw_data()
        data = prepare_data(fraud_df, cc_df)
    else:
        # Load preprocessed data
        logger.info("Loading preprocessed data...")
        data = load_processed_data(resampled=args.use_resampled)
    
    # Filter datasets based on command line argument
    if args.dataset == 'ecommerce':
        return {'ecommerce': data['ecommerce']}
    elif args.dataset == 'creditcard':
        return {'creditcard': data['creditcard']}
    else:
        return data


def get_model_classes(args):
    """
    Get model classes to train based on command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        dict: Dictionary of model classes
    """
    all_models = {
        'logistic': LogisticRegressionModel,
        'rf': RandomForestModel,
        'xgb': XGBoostModel,
        'lgb': LightGBMModel
    }
    
    if args.models == 'all':
        return all_models
    else:
        return {args.models: all_models[args.models]}


def run_training(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Load data
    datasets = prepare_datasets(args)
    
    # Get model classes
    model_classes = get_model_classes(args)
    
    # Create dictionary to store all results
    all_results = []
    
    # Iterate over datasets
    for dataset_name, data in datasets.items():
        logger.info(f"Training models on {dataset_name} dataset...")
        
        # Get training data
        if args.use_resampled and 'X_train_resampled' in data:
            X_train = data['X_train_resampled']
            y_train = data['y_train_resampled']
            logger.info("Using SMOTE resampled training data")
        else:
            X_train = data['X_train']
            y_train = data['y_train']
            logger.info("Using original training data")
            
        X_test = data['X_test']
        y_test = data['y_test']
        
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        logger.info(f"Training labels distribution: {np.bincount(y_train.astype(int))}")
        logger.info(f"Test labels distribution: {np.bincount(y_test.astype(int))}")
        
        # Train and evaluate models
        for model_name, model_class in model_classes.items():
            logger.info(f"Training {model_name} model...")
            
            # Initialize and train model
            model = model_class()
            model.fit(X_train, y_train)
            
            # Save model
            model_path = model.save(dataset_name)
            logger.info(f"Model saved to {model_path}")
            
            # Evaluate model
            logger.info("Evaluating model...")
            metrics, figures = evaluate_and_plot(
                model.model, X_train, y_train, X_test, y_test, dataset_name, model_name
            )
            
            # Save evaluation results
            results_path = save_evaluation_results(metrics, figures)
            logger.info(f"Evaluation results saved to {results_path}")
            
            # Store results for comparison
            all_results.append(metrics)
    
    # Compare models
    logger.info("\nModel Comparison:")
    logger.info("="*60)
    
    # Create a DataFrame for easy comparison
    comparison_df = pd.DataFrame(all_results)
    
    # Select key metrics for comparison
    key_metrics = ['dataset', 'model', 'test_precision', 'test_recall', 'test_f1', 'test_auc', 'test_gmean']
    comparison_df = comparison_df[key_metrics]
    
    # Sort by test F1 score
    comparison_df = comparison_df.sort_values('test_f1', ascending=False)
    
    # Print comparison
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_path = os.path.join(RESULTS_DIR, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Model comparison saved to {comparison_path}")
    
    # Identify best model based on F1 score
    best_model_row = comparison_df.iloc[0]
    logger.info(f"\nBest model: {best_model_row['model']} on {best_model_row['dataset']}")
    logger.info(f"F1 Score: {best_model_row['test_f1']:.4f}")
    logger.info(f"AUC: {best_model_row['test_auc']:.4f}")
    logger.info(f"G-mean: {best_model_row['test_gmean']:.4f}")
    
    return comparison_df


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Fraud Detection Model Training")
    logger.info("="*60)
    
    # Parse command line arguments
    args = parse_args()
    logger.info(f"Arguments: {args}")
    
    # Run training
    results = run_training(args) 