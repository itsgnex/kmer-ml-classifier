#!/usr/bin/env python3
"""
Random Forest training pipeline for species classification.
We use RF as our main model - the ensemble approach handles the 4096-dimensional
k-mer space well and usually outperforms linear models on this type of data.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
    Same 70/15/15 split as other models - ensures fair comparison.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    val_size = 15 / 85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_data(X_train, X_val, X_test):
    """
    Standardize features - even though RF is less sensitive to feature scales,
    keeping the same preprocessing as other models lets us compare fairly.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Data normalized using StandardScaler")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_model(X_train, y_train):
    """
    Train Random Forest classifier.
    Using 100 trees with balanced class weights - this gave us the best results
    during initial experiments. The balanced weights help with the E. coli dominance
    issue (15k+ samples vs 360 for E. ruysiae).
    """
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        n_jobs=-1,  # Use all CPU cores for training speed
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
    Generate confusion matrix from test predictions.
    RF typically shows clean diagonal - but we should check if E. albertii
    and E. coli get confused (they're phylogenetically close).
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
    Helps visualize how well-separated the species clusters are in feature space.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Applying PCA (n_components=50)...")
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_test_scaled)
    logger.info(f"PCA variance explained: {sum(pca.explained_variance_ratio_):.4f}")
    
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
    """Full training pipeline: load, split, train, evaluate, visualize."""
    folder_name = get_dataset_folder()
    
    logger.info(f"Loading data from features_{folder_name}/")
    X, y = load_data(folder_name)
    
    logger.info("Splitting data (70% train, 15% val, 15% test)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    logger.info("Normalizing data...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_data(X_train, X_val, X_test)
    
    logger.info("Training Random Forest model...")
    model = train_model(X_train_scaled, y_train)
    
    logger.info("Evaluating on validation set...")
    val_pred, val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, X_val_scaled, y_val, "Validation")
    
    logger.info("Evaluating on test set...")
    test_pred, test_acc, test_prec, test_rec, test_f1 = evaluate_model(model, X_test_scaled, y_test, "Test")
    
    results_dir = "results_rf"
    os.makedirs(results_dir, exist_ok=True)
    
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
