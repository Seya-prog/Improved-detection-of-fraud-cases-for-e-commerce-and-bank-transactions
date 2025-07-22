# Fraud Detection Models

This project implements machine learning models to detect fraud in e-commerce and bank credit card transactions. The implementation follows industry best practices with a proper Python package structure.

## Project Structure

- `src/`: Source code
  - `data/`: Data loading and preprocessing
  - `models/`: Model definitions
  - `utils/`: Utility functions for evaluation and visualization
  - `train.py`: Main training script
  - `predict.py`: Script for making predictions with trained models
- `data/`: Data files
  - `raw/`: Original datasets
  - `processed/`: Preprocessed datasets
- `models/`: Saved trained models
- `results/`: Evaluation results and visualizations
- `tests/`: Unit tests

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`

## Usage

### 1. Train Models

To train fraud detection models, use the `train.py` script:

```bash
python src/train.py [options]
```

Options:
- `--dataset`: Which dataset to use ('ecommerce', 'creditcard', 'both')
- `--models`: Which models to train ('logistic', 'rf', 'xgb', 'lgb', 'all')
- `--use-resampled`: Use SMOTE resampled data for training
- `--prepare-data`: Prepare data from raw files (if False, load preprocessed data)

Examples:
```bash
# Train all models on all datasets using preprocessed data
python src/train.py

# Train logistic regression on e-commerce data with SMOTE
python src/train.py --dataset ecommerce --models logistic --use-resampled

# Prepare data from raw files and train XGBoost on credit card data
python src/train.py --dataset creditcard --models xgb --prepare-data
```

### 2. Make Predictions

To make predictions with trained models, use the `predict.py` script:

```bash
python src/predict.py --model <model_path> --data <data_path> [options]
```

Required arguments:
- `--model`: Path to the trained model file
- `--data`: Path to the data file to make predictions on

Options:
- `--output`: Path to save the predictions
- `--threshold`: Probability threshold for classification (default: 0.5)

Example:
```bash
python src/predict.py --model models/creditcard_XGBoost_20230701_123456.joblib --data data/processed/creditcard_X_test.npy --threshold 0.3
```

## Model Descriptions

1. **Logistic Regression**
   - Simple, interpretable baseline model
   - Good for understanding feature importance
   - Balances bias and variance for this binary classification task

2. **Random Forest**
   - Ensemble of decision trees to prevent overfitting
   - Handles non-linear relationships and interactions between features
   - Provides feature importance scores

3. **XGBoost**
   - Gradient boosting implementation optimized for performance
   - Handles class imbalance with `scale_pos_weight` parameter
   - Often achieves state-of-the-art results on tabular data

4. **LightGBM**
   - Gradient boosting framework using tree-based learning
   - Fast training speed and low memory usage
   - Good performance on large datasets

## Evaluation Metrics

For imbalanced classification problems like fraud detection, we focus on these metrics:

1. **Precision**: Proportion of true fraud cases among predicted fraud cases
2. **Recall**: Proportion of detected fraud cases among all actual fraud cases
3. **F1-Score**: Harmonic mean of precision and recall
4. **AUC-ROC**: Area under the ROC curve, measuring discrimination
5. **G-mean**: Geometric mean of sensitivity and specificity

## Results

Model evaluation results are saved to the `results/` directory, including:
- Evaluation metrics in JSON format
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Feature importance plots
- Model comparison summary

## Key Findings

When comparing models:
- Evaluate the tradeoff between precision and recall based on business needs
- Consider the cost of false positives vs. false negatives
- Look beyond accuracy due to class imbalance
- AUC-PR (Precision-Recall) is often more informative than AUC-ROC for imbalanced data 