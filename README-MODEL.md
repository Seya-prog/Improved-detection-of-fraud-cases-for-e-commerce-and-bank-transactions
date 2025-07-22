# Fraud Detection Models

**Project Status: Complete**

This project implements machine learning models to detect fraud in e-commerce and bank credit card transactions, following industry best practices and providing full model explainability.

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
- `notebooks/`: EDA, preprocessing, and explainability notebooks

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

### 2. Make Predictions

To make predictions with trained models, use the `predict.py` script:

```bash
python src/predict.py --model <model_path> --data <data_path> [options]
```

### 3. Model Explainability

- See `notebooks/Task3_Model_Explainability.ipynb` for SHAP-based interpretation of the best model.
- SHAP summary, bar, force, and dependence plots are provided for the Random Forest model on the creditcard dataset.

## Model Descriptions

- **Logistic Regression**: Interpretable baseline
- **Random Forest**: Best overall performance (F1: 0.83, AUC: 0.96)
- **XGBoost & LightGBM**: Powerful ensemble alternatives

## Evaluation Metrics

- Precision, Recall, F1-Score, AUC-ROC, G-mean
- All metrics and plots are in the `results/` directory

## Key Findings

- **Best Model**: Random Forest on creditcard dataset
- **Key Features**: Principal components (V14, V17, V12, V10), Amount
- **Explainability**: SHAP analysis reveals the most influential features and their impact on fraud predictions

## Project Completion

All tasks are complete, including robust CI/CD, modular code, and full model explainability. See the main `README.md` for a project summary and quickstart. 