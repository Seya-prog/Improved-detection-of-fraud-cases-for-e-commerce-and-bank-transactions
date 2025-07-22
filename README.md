# Improved Fraud Detection for E-commerce and Bank Transactions

**Project Status: Complete**

This project delivers a robust, end-to-end fraud detection pipeline for e-commerce and bank credit transactions. It includes data analysis, model building, evaluation, and model explainability using SHAP.

## Project Structure
- `data/`: Datasets (Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv)
- `src/`: Source code for data processing, feature engineering, model building, and utilities
- `notebooks/`: Jupyter notebooks for EDA, preprocessing, and model explainability
- `models/`: Saved trained models and metadata
- `results/`: Evaluation metrics, plots, and SHAP explainability outputs
- `tests/`: Unit and integration tests

## Datasets
1. **Fraud_Data.csv**: E-commerce transaction data
2. **IpAddress_to_Country.csv**: Maps IP addresses to countries
3. **creditcard.csv**: Bank transaction data

## Quickstart
1. Clone this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```
3. Train models (using preprocessed data):
```bash
python src/train.py --use-resampled
```
4. Run model explainability (see `notebooks/Task3_Model_Explainability.ipynb`)

## Project Tasks
1. **Data Analysis and Preprocessing**: Cleaning, feature engineering, and transformation
2. **Model Building and Training**: Logistic Regression (baseline) and Ensemble Models (Random Forest, XGBoost, LightGBM)
3. **Model Evaluation and Selection**: Imbalanced metrics (AUC-PR, F1, G-mean, Confusion Matrix)
4. **Model Explainability**: SHAP analysis for best-performing model

## Results and Interpretation
- **Best Model**: Random Forest on the creditcard dataset (F1: 0.83, AUC: 0.96)
- **Key Features**: Principal components (V14, V17, V12, V10), Amount
- **Explainability**: SHAP plots reveal which features drive fraud predictions
- **Artifacts**: All results, plots, and model files are in the `results/` and `models/` directories

## Critical Challenges
- Handling highly imbalanced datasets
- Balancing security with user experience (minimizing false positives and negatives)
- Providing interpretable, actionable model outputs for business use 
- Balancing security with user experience (minimizing false positives and negatives) 