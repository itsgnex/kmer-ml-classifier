#!/usr/bin/env python3
"""
Script to generate t-SNE scatter plots from pre-computed embeddings.
Loads embedding_data.csv from result folders and creates publication-ready plots.
This is useful when we want to visualize model-specific clustering patterns.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_unique_labels(labels):
    """Extract and sort unique labels for consistent color mapping."""
    return sorted(labels.unique())


def create_tsne_plot(csv_path, output_path, dataset_type, run_num, model_name):
    """
    Create a publication-quality t-SNE scatter plot from embedding data.
    Loads the CSV with t-SNE coordinates and species labels, then plots each species.
    """
    df = pd.read_csv(csv_path)
    
    unique_labels = get_unique_labels(df['label'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for idx, label in enumerate(unique_labels):
        mask = df['label'] == label
        ax.scatter(
            df.loc[mask, 'x_tsne'],
            df.loc[mask, 'y_tsne'],
            c=[colors[idx]],
            label=label,
            alpha=0.6,
            s=50,
            edgecolors='none'
        )
    
    ax.set_xlabel('t-SNE 1', fontsize=12, labelpad=10)
    ax.set_ylabel('t-SNE 2', fontsize=12, labelpad=10)
    ax.set_title(f't-SNE Visualization - {dataset_type} Results\nRun {run_num} - Model: {model_name}', 
                 fontsize=14, pad=15)
    
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
    """
    Main function to generate t-SNE plots for selected results directory.
    Traverses through Old_results or New_results, finds all embedding_data.csv files,
    and generates plots for each run/model combination.
    """
    root_dir = os.getcwd()
    
    print("=" * 60)
    print("t-SNE Plot Generator")
    print("=" * 60)
    print("\nSelect results type:")
    print("  'old' - Use Old_results directory")
    print("  'new' - Use New_results directory")
    
    user_input = input("\nEnter 'old' or 'new': ").strip().lower()
    
    if user_input == 'old':
        base_dir = 'Old_results'
        dataset_type = 'OLD'
    elif user_input == 'new':
        base_dir = 'New_results'
        dataset_type = 'NEW'
    else:
        print("Error: Invalid input. Please enter 'old' or 'new'.")
        return
    
    base_path = os.path.join(root_dir, base_dir)
    
    if not os.path.exists(base_path):
        print(f"Error: Directory not found: {base_path}")
        return
    
    print(f"\nUsing directory: {base_path}")
    print("-" * 60)
    
    # Look for runs 1, 2, 3 and models lr, rf, svm_rbf
    run_numbers = ['1', '2', '3']
    model_folders = ['results_lr', 'results_rf', 'results_svm_rbf']
    
    model_names = {
        'results_lr': 'Logistic Regression',
        'results_rf': 'Random Forest',
        'results_svm_rbf': 'SVM (RBF Kernel)'
    }
    
    plot_count = 0
    skipped_count = 0
    
    for run_num in run_numbers:
        print(f"\nProcessing Run {run_num}:")
        
        for model_folder in model_folders:
            embedding_file = 'embedding_data.csv'
            csv_path = os.path.join(base_path, run_num, model_folder, embedding_file)
            
            if os.path.exists(csv_path):
                output_path = os.path.join(base_path, run_num, model_folder, 'tsne_plot.png')
                
                model_display_name = model_names.get(model_folder, model_folder)
                create_tsne_plot(csv_path, output_path, dataset_type, run_num, model_display_name)
                plot_count += 1
            else:
                print(f"  Skipping {model_folder}: embedding_data.csv not found")
                skipped_count += 1
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Plots generated: {plot_count}")
    print(f"  Folders skipped: {skipped_count}")
    print("=" * 60)


if __name__ == '__main__':
    main()
