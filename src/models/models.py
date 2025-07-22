"""
Models for fraud detection.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import numpy as np
import json
from datetime import datetime


# Define models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')


class LogisticRegressionModel:
    """Logistic Regression baseline model for fraud detection."""
    
    def __init__(self, max_iter=1000, C=1.0, class_weight='balanced', random_state=42):
        """
        Initialize the logistic regression model.
        
        Args:
            max_iter (int): Maximum number of iterations
            C (float): Inverse of regularization strength
            class_weight (str): Weight for classes
            random_state (int): Random state for reproducibility
        """
        self.model = LogisticRegression(
            max_iter=max_iter,
            C=C,
            class_weight=class_weight,
            random_state=random_state
        )
        self.name = "LogisticRegression"
        self.hyperparams = {
            'max_iter': max_iter,
            'C': C,
            'class_weight': class_weight,
            'random_state': random_state
        }
        
    def fit(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            self: Trained model
        """
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            array: Predicted labels
        """
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            array: Class probabilities
        """
        return self.model.predict_proba(X)
        
    def save(self, dataset_name):
        """
        Save model to disk.
        
        Args:
            dataset_name (str): Name of the dataset the model was trained on
        
        Returns:
            str: Path to saved model
        """
        os.makedirs(MODELS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{dataset_name}_{self.name}_{timestamp}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        
        # Save the model
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'model_type': self.name,
            'dataset': dataset_name,
            'timestamp': timestamp,
            'hyperparameters': self.hyperparams,
            'feature_importance': None
        }
        
        # Add feature importance if available
        if hasattr(self.model, 'coef_'):
            metadata['feature_importance'] = self.model.coef_[0].tolist()
        
        metadata_filename = f"{dataset_name}_{self.name}_{timestamp}_metadata.json"
        metadata_path = os.path.join(MODELS_DIR, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")
        
        return model_path


class RandomForestModel:
    """Random Forest model for fraud detection."""
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 class_weight='balanced', random_state=42):
        """
        Initialize the random forest model.
        
        Args:
            n_estimators (int): Number of trees
            max_depth (int): Maximum depth of trees
            min_samples_split (int): Minimum samples required to split
            class_weight (str): Weight for classes
            random_state (int): Random state for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=random_state
        )
        self.name = "RandomForest"
        self.hyperparams = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'class_weight': class_weight,
            'random_state': random_state
        }
        
    def fit(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            self: Trained model
        """
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            array: Predicted labels
        """
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            array: Class probabilities
        """
        return self.model.predict_proba(X)
        
    def save(self, dataset_name):
        """
        Save model to disk.
        
        Args:
            dataset_name (str): Name of the dataset the model was trained on
        
        Returns:
            str: Path to saved model
        """
        os.makedirs(MODELS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{dataset_name}_{self.name}_{timestamp}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        
        # Save the model
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'model_type': self.name,
            'dataset': dataset_name,
            'timestamp': timestamp,
            'hyperparameters': self.hyperparams,
            'feature_importance': None
        }
        
        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            metadata['feature_importance'] = self.model.feature_importances_.tolist()
        
        metadata_filename = f"{dataset_name}_{self.name}_{timestamp}_metadata.json"
        metadata_path = os.path.join(MODELS_DIR, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")
        
        return model_path


class XGBoostModel:
    """XGBoost model for fraud detection."""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, 
                 scale_pos_weight=None, random_state=42):
        """
        Initialize the XGBoost model.
        
        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum depth of trees
            learning_rate (float): Learning rate
            scale_pos_weight (float): Scale for positive class weight
            random_state (int): Random state for reproducibility
        """
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state
        )
        self.name = "XGBoost"
        self.hyperparams = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'scale_pos_weight': scale_pos_weight,
            'random_state': random_state
        }
        
    def fit(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            self: Trained model
        """
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            array: Predicted labels
        """
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            array: Class probabilities
        """
        return self.model.predict_proba(X)
        
    def save(self, dataset_name):
        """
        Save model to disk.
        
        Args:
            dataset_name (str): Name of the dataset the model was trained on
        
        Returns:
            str: Path to saved model
        """
        os.makedirs(MODELS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{dataset_name}_{self.name}_{timestamp}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        
        # Save the model
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'model_type': self.name,
            'dataset': dataset_name,
            'timestamp': timestamp,
            'hyperparameters': self.hyperparams,
            'feature_importance': None
        }
        
        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            metadata['feature_importance'] = self.model.feature_importances_.tolist()
        
        metadata_filename = f"{dataset_name}_{self.name}_{timestamp}_metadata.json"
        metadata_path = os.path.join(MODELS_DIR, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")
        
        return model_path


class LightGBMModel:
    """LightGBM model for fraud detection."""
    
    def __init__(self, n_estimators=100, max_depth=-1, learning_rate=0.1, 
                 class_weight=None, random_state=42):
        """
        Initialize the LightGBM model.
        
        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum depth of trees
            learning_rate (float): Learning rate
            class_weight (dict): Weight for classes
            random_state (int): Random state for reproducibility
        """
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            class_weight=class_weight,
            random_state=random_state
        )
        self.name = "LightGBM"
        self.hyperparams = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'class_weight': class_weight,
            'random_state': random_state
        }
        
    def fit(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            self: Trained model
        """
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            array: Predicted labels
        """
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            array: Class probabilities
        """
        return self.model.predict_proba(X)
        
    def save(self, dataset_name):
        """
        Save model to disk.
        
        Args:
            dataset_name (str): Name of the dataset the model was trained on
        
        Returns:
            str: Path to saved model
        """
        os.makedirs(MODELS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{dataset_name}_{self.name}_{timestamp}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        
        # Save the model
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'model_type': self.name,
            'dataset': dataset_name,
            'timestamp': timestamp,
            'hyperparameters': self.hyperparams,
            'feature_importance': None
        }
        
        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            metadata['feature_importance'] = self.model.feature_importances_.tolist()
        
        metadata_filename = f"{dataset_name}_{self.name}_{timestamp}_metadata.json"
        metadata_path = os.path.join(MODELS_DIR, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")
        
        return model_path 