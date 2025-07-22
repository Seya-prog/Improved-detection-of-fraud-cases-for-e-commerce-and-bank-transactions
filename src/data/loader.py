"""
Data loading and preprocessing functionality for fraud detection models.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Define data directories
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')


def load_raw_data():
    """
    Load raw data from files.
    
    Returns:
        tuple: (fraud_df, cc_df) - E-commerce fraud data and credit card fraud data
    """
    print("Loading raw data...")
    fraud_df = pd.read_csv(os.path.join(RAW_DIR, 'Fraud_Data.csv'))
    cc_df = pd.read_csv(os.path.join(RAW_DIR, 'creditcard.csv'))
    
    print(f"Loaded Fraud_Data.csv with {fraud_df.shape[0]} rows and {fraud_df.shape[1]} columns")
    print(f"Loaded creditcard.csv with {cc_df.shape[0]} rows and {cc_df.shape[1]} columns")
    
    return fraud_df, cc_df


def load_processed_data(resampled=True):
    """
    Load processed data ready for model training.
    
    Args:
        resampled (bool): Whether to load the SMOTE-resampled data
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) for both datasets as a dictionary
    """
    print("Loading processed data...")
    data = {}
    
    # Load e-commerce data
    prefix = 'ecommerce'
    suffix = '_resampled' if resampled else ''
    
    data[prefix] = {
        'X_train': np.load(os.path.join(PROCESSED_DIR, 'model_ready', f'{prefix}_X_train{suffix}.npy'), allow_pickle=True),
        'X_test': np.load(os.path.join(PROCESSED_DIR, 'model_ready', f'{prefix}_X_test.npy'), allow_pickle=True),
        'y_train': np.load(os.path.join(PROCESSED_DIR, 'model_ready', f'{prefix}_y_train{suffix}.npy'), allow_pickle=True),
        'y_test': np.load(os.path.join(PROCESSED_DIR, 'model_ready', f'{prefix}_y_test.npy'), allow_pickle=True)
    }
    
    # Load credit card data
    prefix = 'creditcard'
    data[prefix] = {
        'X_train': np.load(os.path.join(PROCESSED_DIR, 'model_ready', f'{prefix}_X_train{suffix}.npy'), allow_pickle=True),
        'X_test': np.load(os.path.join(PROCESSED_DIR, 'model_ready', f'{prefix}_X_test.npy'), allow_pickle=True),
        'y_train': np.load(os.path.join(PROCESSED_DIR, 'model_ready', f'{prefix}_y_train{suffix}.npy'), allow_pickle=True),
        'y_test': np.load(os.path.join(PROCESSED_DIR, 'model_ready', f'{prefix}_y_test.npy'), allow_pickle=True)
    }
    
    print("Processed data loaded successfully!")
    return data


def prepare_data(fraud_df, cc_df, test_size=0.3, random_state=42):
    """
    Prepare data for modeling by cleaning, preprocessing, and splitting.
    
    Args:
        fraud_df (DataFrame): E-commerce fraud data
        cc_df (DataFrame): Credit card fraud data
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing train/test splits for both datasets
    """
    print("Preparing data for modeling...")
    data = {}
    
    # Process E-commerce fraud data
    print("\nProcessing e-commerce data...")
    
    # Convert datetime columns
    for col in ['signup_time', 'purchase_time']:
        if fraud_df[col].dtype == 'object':
            fraud_df[col] = pd.to_datetime(fraud_df[col])
    
    # Feature engineering
    # Time difference between signup and purchase
    fraud_df['time_since_signup'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds() / 3600
    
    # One-hot encode categorical variables
    categorical_cols = ['sex', 'browser', 'source']
    if 'country' in fraud_df.columns:
        categorical_cols.append('country')
        
    fraud_df_encoded = pd.get_dummies(fraud_df, columns=categorical_cols, drop_first=True)
    
    # Drop columns not needed for modeling
    cols_to_drop = ['user_id', 'device_id', 'ip_address', 'signup_time', 'purchase_time']
    if 'prev_purchase_time' in fraud_df_encoded.columns:
        cols_to_drop.append('prev_purchase_time')
        
    X_ecommerce = fraud_df_encoded.drop(['class'] + cols_to_drop, axis=1, errors='ignore')
    y_ecommerce = fraud_df_encoded['class']
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = X_ecommerce.select_dtypes(include=['int64', 'float64']).columns
    X_ecommerce[numerical_cols] = scaler.fit_transform(X_ecommerce[numerical_cols])
    
    # Process Credit Card data
    print("\nProcessing credit card data...")
    
    # The credit card data is already preprocessed with PCA
    # We just need to scale the 'Amount' feature
    X_cc = cc_df.drop('Class', axis=1)
    y_cc = cc_df['Class']
    
    # Scale Amount
    amount_scaler = StandardScaler()
    X_cc['Amount'] = amount_scaler.fit_transform(X_cc[['Amount']])
    
    # Split the data
    print("\nSplitting data into train and test sets...")
    
    # Split e-commerce data
    X_train_ec, X_test_ec, y_train_ec, y_test_ec = train_test_split(
        X_ecommerce, y_ecommerce, test_size=test_size, random_state=random_state, stratify=y_ecommerce
    )
    
    # Split credit card data
    X_train_cc, X_test_cc, y_train_cc, y_test_cc = train_test_split(
        X_cc, y_cc, test_size=test_size, random_state=random_state, stratify=y_cc
    )
    
    # Apply SMOTE to handle class imbalance
    print("\nApplying SMOTE to handle class imbalance...")
    smote = SMOTE(sampling_strategy=0.5, random_state=random_state)
    
    # For e-commerce data
    X_train_ec_resampled, y_train_ec_resampled = smote.fit_resample(X_train_ec, y_train_ec)
    print(f"E-commerce: Original class distribution: {np.bincount(y_train_ec.astype(int))}")
    print(f"E-commerce: Resampled class distribution: {np.bincount(y_train_ec_resampled.astype(int))}")
    
    # For credit card data
    X_train_cc_resampled, y_train_cc_resampled = smote.fit_resample(X_train_cc, y_train_cc)
    print(f"Credit Card: Original class distribution: {np.bincount(y_train_cc.astype(int))}")
    print(f"Credit Card: Resampled class distribution: {np.bincount(y_train_cc_resampled.astype(int))}")
    
    # Save data to processed directory
    print("\nSaving processed data...")
    os.makedirs(os.path.join(PROCESSED_DIR, 'model_ready'), exist_ok=True)
    
    # Save e-commerce data
    np.save(os.path.join(PROCESSED_DIR, 'model_ready', 'ecommerce_X_train.npy'), X_train_ec)
    np.save(os.path.join(PROCESSED_DIR, 'model_ready', 'ecommerce_X_test.npy'), X_test_ec)
    np.save(os.path.join(PROCESSED_DIR, 'model_ready', 'ecommerce_y_train.npy'), y_train_ec)
    np.save(os.path.join(PROCESSED_DIR, 'model_ready', 'ecommerce_y_test.npy'), y_test_ec)
    np.save(os.path.join(PROCESSED_DIR, 'model_ready', 'ecommerce_X_train_resampled.npy'), X_train_ec_resampled)
    np.save(os.path.join(PROCESSED_DIR, 'model_ready', 'ecommerce_y_train_resampled.npy'), y_train_ec_resampled)
    
    # Save credit card data
    np.save(os.path.join(PROCESSED_DIR, 'model_ready', 'creditcard_X_train.npy'), X_train_cc)
    np.save(os.path.join(PROCESSED_DIR, 'model_ready', 'creditcard_X_test.npy'), X_test_cc)
    np.save(os.path.join(PROCESSED_DIR, 'model_ready', 'creditcard_y_train.npy'), y_train_cc)
    np.save(os.path.join(PROCESSED_DIR, 'model_ready', 'creditcard_y_test.npy'), y_test_cc)
    np.save(os.path.join(PROCESSED_DIR, 'model_ready', 'creditcard_X_train_resampled.npy'), X_train_cc_resampled)
    np.save(os.path.join(PROCESSED_DIR, 'model_ready', 'creditcard_y_train_resampled.npy'), y_train_cc_resampled)
    
    # Create dictionary with the data
    data['ecommerce'] = {
        'X_train': X_train_ec,
        'X_test': X_test_ec,
        'y_train': y_train_ec,
        'y_test': y_test_ec,
        'X_train_resampled': X_train_ec_resampled,
        'y_train_resampled': y_train_ec_resampled
    }
    
    data['creditcard'] = {
        'X_train': X_train_cc,
        'X_test': X_test_cc,
        'y_train': y_train_cc,
        'y_test': y_test_cc,
        'X_train_resampled': X_train_cc_resampled,
        'y_train_resampled': y_train_cc_resampled
    }
    
    print("Data preparation completed successfully!")
    return data


if __name__ == "__main__":
    # For testing
    fraud_df, cc_df = load_raw_data()
    prepare_data(fraud_df, cc_df) 