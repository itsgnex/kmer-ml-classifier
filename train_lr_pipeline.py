#!/usr/bin/env python3
"""
Logistic Regression training pipeline for species classification.
We use LR as a baseline linear model - it's interpretable and gives us a sense
of how linearly separable the 6-mer feature space is.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_dataset_folder():
    """Prompt user for dataset folder."""
    return input("Enter dataset folder (old or new): ").strip()


def load_data(folder_name):
    """Load features and labels from specified folder."""
    if folder_name not in ['old', 'new']:
        logger.error("Invalid folder name. Use 'old' or 'new'.")
        exit(1)
    
    features_dir = f"features_{folder_name}"
    X_path = os.path.join(features_dir, 'X_features.npy')
    y_path = os.path.join(features_dir, 'y_labels.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        logger.error(f"Files not found in {features_dir}")
        exit(1)
    
    X = np.load(X_path)
    y = np.load(y_path)
    
    logger.info(f"Loaded data: X shape = {X.shape}, y shape = {y.shape}")
    return X, y


def split_data(X, y):
    """
    Split data into train/validation/test sets.
    Using 70/15/15 split with stratification to preserve class distribution.
    """
    # First split: 85% temp, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Second split: 70% train, 15% val (15/85 = ~0.176 of original)
    val_size = 15 / 85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_data(X_train, X_val, X_test):
    """
    Standardize features using training set statistics only.
    This prevents data leakage - validation and test should never influence feature scaling.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Data normalized using StandardScaler")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_model(X_train, y_train):
    """
    Train Logistic Regression with class balancing.
    Using balanced weights because our species have very different sample sizes
    (e.g., 15861 E. coli vs 360 E. ruysiae). Without this, the model would just
    predict the majority class.
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    logger.info("Model trained successfully")
    return model


def evaluate_model(model, X, y, dataset_name):
    """Evaluate model and print metrics."""
    y_pred = model.predict(X)
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    
    logger.info(f"\n{dataset_name} Set Metrics:")
    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    return y_pred, acc, prec, rec, f1


def save_confusion_matrix(y_true, y_pred, save_path):
    """
    Generate and save confusion matrix plot.
    We use test set predictions here to see how the model generalizes.
    This helps identify which species are being confused with each other.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")


def compute_embeddings(X_test_scaled, y_test, output_dir):
    """
    Generate PCA and t-SNE embeddings for visualization.
    Important: These are for visualization only, NOT used during training.
    Helps us see if the feature space naturally clusters by species.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Applying PCA (n_components=50)...")
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_test_scaled)
    logger.info(f"PCA variance explained: {sum(pca.explained_variance_ratio_):.4f}")
    
    # PCA 50D first to reduce noise before t-SNE - standard practice
    logger.info("Applying t-SNE (n_components=2)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_pca)-1))
    X_tsne = tsne.fit_transform(X_pca)
    
    df = pd.DataFrame({
        'x_tsne': X_tsne[:, 0],
        'y_tsne': X_tsne[:, 1],
        'label': y_test
    })
    
    csv_path = os.path.join(output_dir, 'embedding_data.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Embedding data saved to {csv_path}")
    
    return df


def save_metrics(metrics_dict, save_path):
    """Save metrics to text file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        for dataset, metrics in metrics_dict.items():
            f.write(f"\n{dataset} Set:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score:  {metrics['f1']:.4f}\n")
    
    logger.info(f"Metrics saved to {save_path}")


def main():
    """Full training pipeline: load features, split, train, evaluate."""
    folder_name = get_dataset_folder()
    
    logger.info(f"Loading data from features_{folder_name}/")
    X, y = load_data(folder_name)
    
    logger.info("Splitting data (70% train, 15% val, 15% test)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    logger.info("Normalizing data...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_data(X_train, X_val, X_test)
    
    logger.info("Training Logistic Regression model...")
    model = train_model(X_train_scaled, y_train)
    
    logger.info("Evaluating on validation set...")
    val_pred, val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, X_val_scaled, y_val, "Validation")
    
    logger.info("Evaluating on test set...")
    test_pred, test_acc, test_prec, test_rec, test_f1 = evaluate_model(model, X_test_scaled, y_test, "Test")
    
    results_dir = "results_lr"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save test set confusion matrix for paper figures
    save_confusion_matrix(y_test, test_pred, os.path.join(results_dir, 'confusion_matrix.png'))
    
    metrics_dict = {
        'Validation': {'accuracy': val_acc, 'precision': val_prec, 'recall': val_rec, 'f1': val_f1},
        'Test': {'accuracy': test_acc, 'precision': test_prec, 'recall': test_rec, 'f1': test_f1}
    }
    save_metrics(metrics_dict, os.path.join(results_dir, 'metrics.txt'))
    
    logger.info("Generating PCA and t-SNE embeddings...")
    compute_embeddings(X_test_scaled, y_test, results_dir)
    
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
