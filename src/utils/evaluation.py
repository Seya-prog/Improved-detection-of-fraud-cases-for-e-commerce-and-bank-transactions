"""
Evaluation utilities for fraud detection models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, precision_recall_curve, roc_curve, auc,
    confusion_matrix, classification_report
)
from imblearn.metrics import geometric_mean_score
import os
import json
from datetime import datetime


# Define results directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')


def evaluate_model(model, X_train, y_train, X_test, y_test, dataset_name, model_name):
    """
    Evaluate a model on train and test sets.
    
    Args:
        model: Trained model with predict and predict_proba methods
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        dataset_name: Name of the dataset
        model_name: Name of the model
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Get probability predictions
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'dataset': dataset_name,
        'model': model_name,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'train_gmean': geometric_mean_score(y_train, y_train_pred),
        'test_gmean': geometric_mean_score(y_test, y_test_pred),
        'train_auc': roc_auc_score(y_train, y_train_prob),
        'test_auc': roc_auc_score(y_test, y_test_prob),
    }
    
    return metrics


def print_evaluation_report(metrics):
    """
    Print evaluation metrics in a readable format.
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    print(f"\nModel Evaluation: {metrics['model']} on {metrics['dataset']} Dataset")
    print("="*60)
    print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Train Precision: {metrics['train_precision']:.4f}")
    print(f"Test Precision: {metrics['test_precision']:.4f}")
    print(f"Train Recall: {metrics['train_recall']:.4f}")
    print(f"Test Recall: {metrics['test_recall']:.4f}")
    print(f"Train F1: {metrics['train_f1']:.4f}")
    print(f"Test F1: {metrics['test_f1']:.4f}")
    print(f"Train G-mean: {metrics['train_gmean']:.4f}")
    print(f"Test G-mean: {metrics['test_gmean']:.4f}")
    print(f"Train AUC: {metrics['train_auc']:.4f}")
    print(f"Test AUC: {metrics['test_auc']:.4f}")


def save_evaluation_results(metrics, figures=None):
    """
    Save evaluation results to disk.
    
    Args:
        metrics: Dictionary of evaluation metrics
        figures: List of (figure, filename) tuples to save
        
    Returns:
        str: Path to saved results
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics
    results_filename = f"{metrics['dataset']}_{metrics['model']}_{timestamp}_results.json"
    results_path = os.path.join(RESULTS_DIR, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save figures if provided
    if figures:
        figures_dir = os.path.join(RESULTS_DIR, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        for fig, filename in figures:
            fig_path = os.path.join(figures_dir, f"{metrics['dataset']}_{metrics['model']}_{timestamp}_{filename}")
            fig.savefig(fig_path, bbox_inches='tight', dpi=300)
    
    print(f"Evaluation results saved to {results_path}")
    return results_path


def plot_confusion_matrix(y_true, y_pred, title=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Title for the plot
        
    Returns:
        fig: Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(title if title else 'Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_prob, title=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Title for the plot
        
    Returns:
        fig: Matplotlib figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title if title else 'Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(y_true, y_prob, title=None):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Title for the plot
        
    Returns:
        fig: Matplotlib figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title if title else 'Precision-Recall Curve')
    ax.legend(loc="lower left")
    plt.tight_layout()
    return fig


def evaluate_and_plot(model, X_train, y_train, X_test, y_test, dataset_name, model_name):
    """
    Evaluate a model and create plots.
    
    Args:
        model: Trained model with predict and predict_proba methods
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        dataset_name: Name of the dataset
        model_name: Name of the model
        
    Returns:
        tuple: (metrics, figures)
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, dataset_name, model_name)
    print_evaluation_report(metrics)
    
    # Create figures
    figures = []
    
    # Confusion Matrix
    cm_train_fig = plot_confusion_matrix(
        y_train, y_train_pred, 
        title=f'{model_name} - Training Confusion Matrix'
    )
    figures.append((cm_train_fig, 'train_confusion_matrix.png'))
    
    cm_test_fig = plot_confusion_matrix(
        y_test, y_test_pred, 
        title=f'{model_name} - Test Confusion Matrix'
    )
    figures.append((cm_test_fig, 'test_confusion_matrix.png'))
    
    # ROC Curve
    roc_test_fig = plot_roc_curve(
        y_test, y_test_prob, 
        title=f'{model_name} - ROC Curve'
    )
    figures.append((roc_test_fig, 'test_roc_curve.png'))
    
    # Precision-Recall Curve
    pr_test_fig = plot_precision_recall_curve(
        y_test, y_test_prob, 
        title=f'{model_name} - Precision-Recall Curve'
    )
    figures.append((pr_test_fig, 'test_pr_curve.png'))
    
    return metrics, figures 