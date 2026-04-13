#!/usr/bin/env python3
"""
Script to generate PCA and t-SNE visualizations from k-mer features.
This helps us visually verify whether species separate in the 6-mer feature space.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_label_mapping(folder_path):
    """
    Load label mapping from label_mapping.txt if available.
    This lets us show species names instead of just numbers in the legend.
    """
    mapping_path = os.path.join(folder_path, 'label_mapping.txt')
    
    if not os.path.exists(mapping_path):
        print(f"  Warning: label_mapping.txt not found in {folder_path}")
        return None
    
    label_map = {}
    try:
        with open(mapping_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    key, value = line.split(':', 1)
                    label_map[int(key.strip())] = value.strip()
        print(f"  Loaded label mapping with {len(label_map)} classes")
        return label_map
    except Exception as e:
        print(f"  Warning: Could not parse label_mapping.txt: {e}")
        return None


def get_label_names(labels, label_map):
    """Convert numeric labels to species names if mapping is available."""
    if label_map is None:
        return [f"Class {int(l)}" for l in labels]
    return [label_map.get(int(l), f"Class {int(l)}") for l in labels]


def create_scatter_plot(x, y, labels, output_path, title, xlabel, ylabel, label_map):
    """
    Create a publication-quality scatter plot.
    Using enough colors to distinguish up to 20 species, with clear legends.
    """
    unique_labels = np.unique(labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use tab20 colormap - enough colors for most datasets
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        label_name = label_map.get(int(label), f"Class {int(label)}") if label_map else f"Class {int(label)}"
        
        ax.scatter(
            x[mask],
            y[mask],
            c=[colors[idx]],
            label=label_name,
            alpha=0.6,
            s=50,
            edgecolors='none'
        )
    
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, pad=15)
    
    # Place legend outside the plot for clarity
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=9,
        title='Species',
        title_fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Plot saved: {output_path}")


def main():
    """Main function to generate PCA and t-SNE plots."""
    root_dir = os.getcwd()
    
    print("=" * 60)
    print("PCA and t-SNE Plot Generator")
    print("=" * 60)
    print("\nSelect dataset:")
    print("  'old' - Use features_old directory")
    print("  'new' - Use features_new directory")
    
    user_input = input("\nEnter 'old' or 'new': ").strip().lower()
    
    if user_input == 'old':
        dataset_dir = 'features_old'
        dataset_name = 'OLD'
    elif user_input == 'new':
        dataset_dir = 'features_new'
        dataset_name = 'NEW'
    else:
        print("Error: Invalid input. Please enter 'old' or 'new'.")
        return
    
    folder_path = os.path.join(root_dir, dataset_dir)
    
    if not os.path.exists(folder_path):
        print(f"Error: Directory not found: {folder_path}")
        return
    
    print(f"\nUsing dataset: {dataset_dir}")
    print("-" * 60)
    
    features_path = os.path.join(folder_path, 'X_features.npy')
    labels_path = os.path.join(folder_path, 'y_labels.npy')
    
    if not os.path.exists(features_path):
        print(f"Error: X_features.npy not found in {folder_path}")
        return
    
    if not os.path.exists(labels_path):
        print(f"Error: y_labels.npy not found in {folder_path}")
        return
    
    print("\nLoading data...")
    X = np.load(features_path)
    y = np.load(labels_path)
    
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Number of classes: {len(np.unique(y))}")
    
    label_map = load_label_mapping(folder_path)
    
    print("\n" + "-" * 60)
    print("Computing PCA (2D)...")
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X)
    
    explained_var = pca_2d.explained_variance_ratio_
    print(f"  Explained variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}")
    
    pca_output = os.path.join(folder_path, 'pca_plot.png')
    create_scatter_plot(
        X_pca_2d[:, 0], X_pca_2d[:, 1], y,
        pca_output,
        f'PCA Visualization - {dataset_name} Dataset',
        'PC1', 'PC2', label_map
    )
    
    print("\n" + "-" * 60)
    print("Computing PCA (50D) for t-SNE...")
    # PCA to 50D first reduces noise and speeds up t-SNE significantly
    pca_50d = PCA(n_components=50)
    X_pca_50d = pca_50d.fit_transform(X)
    
    retained_var = sum(pca_50d.explained_variance_ratio_)
    print(f"  Retained variance with 50 PCs: {retained_var:.2%}")
    
    print("\nComputing t-SNE (2D)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_pca_50d)
    
    tsne_output = os.path.join(folder_path, 'tsne_plot.png')
    create_scatter_plot(
        X_tsne[:, 0], X_tsne[:, 1], y,
        tsne_output,
        f't-SNE Visualization - {dataset_name} Dataset (PCA→t-SNE)',
        't-SNE 1', 't-SNE 2', label_map
    )
    
    print("\n" + "=" * 60)
    print("Completed!")
    print(f"  PCA plot: {pca_output}")
    print(f"  t-SNE plot: {tsne_output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
