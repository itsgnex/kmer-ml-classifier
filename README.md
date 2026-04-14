# Species-Level Classification of Closely Related Bacteria Using Alignment-Free DNA Features and Machine Learning

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit-learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A bioinformatics machine learning project demonstrating species-level classification of closely related bacteria using alignment-free 6-mer DNA features and supervised machine learning.

---

## Key Highlights

- **Alignment-Free Classification**: Uses k-mer (6-mer) frequency features instead of sequence alignment
- **Biological Resolution Boundary**: Evaluates whether short 16S rRNA sequences can distinguish same-genus species
- **Three ML Models**: Logistic Regression, Random Forest, and SVM with RBF kernel
- **Class Imbalance Handling**: Weighted loss functions with `class_weight='balanced'`
- **Dimensionality Reduction**: PCA and t-SNE visualizations for cluster analysis
- **Publication-Ready Results**: 98.04% test accuracy with Random Forest

---

## Research Objective

This project investigates the **biological resolution boundary** of short 16S rRNA sequences for species-level classification in closely related bacteria. The key question: *Can alignment-free DNA k-mer features distinguish between same-genus species that share high sequence similarity?*

---

## Biological Motivation

The 16S rRNA gene is a standard marker for bacterial taxonomy. However, same-genus species (e.g., *E. coli* vs. *E. albertii*) share >95% sequence identity, making classification challenging. This project evaluates whether:

1. 6-mer k-mer frequency features capture species-specific patterns
2. Machine learning can learn discriminative signatures without alignment
3. Short amplicons retain sufficient phylogenetic signal for species differentiation

---

## Dataset Summary

### Two Datasets Used

| Dataset | Species | Biological Context |
|---------|---------|---------------------|
| **OLD** | 5 *Escherichia* species | Closely related enteropathogens |
| **NEW** | 4 *Shigella* + 1 *Salmonella* | Enteric pathogens from related genera |

### OLD Dataset (Enteropathogenic *Escherichia*)

- *E. marmotae* (1,255 sequences)
- *E. coli* (15,861 sequences)
- *E. ruysiae* (360 sequences)
- *E. fergusoni* (3,865 sequences)
- *E. albertii* (8,003 sequences)
- **Total: 29,344 sequences**

### NEW Dataset (Enteric Pathogens)

- *Shigella flexneri* (1,248 sequences)
- *Shigella dysenteriae* (1,279 sequences)
- *Shigella sonnei* (138 sequences)
- *Salmonella enterica* (49 sequences)
- *Shigella boydii* (900 sequences)
- **Total: 3,614 sequences**

---

## Repository Structure

```
v3-v4/
├── features_old/                          # Extracted features (OLD)
│   ├── X_features.npy                     # (29344, 4096) float32
│   ├── y_labels.npy                       # (29344,) int32
│   ├── label_mapping.txt                  # Species name mapping
│   ├── pca_plot.png                       # PCA 2D visualization
│   └── tsne_plot.png                      # t-SNE visualization
├── features_new/                          # Extracted features (NEW)
│   ├── X_features.npy                     # (3614, 4096) float32
│   ├── y_labels.npy                       # (3614,) int32
│   ├── label_mapping.txt                  # Species name mapping
│   ├── pca_plot.png                       # PCA 2D visualization
│   └── tsne_plot.png                      # t-SNE visualization
├── Old_results/                           # Training results (OLD dataset)
│   ├── 1/
│   │   ├── results_lr/
│   │   │   ├── metrics.txt
│   │   │   ├── confusion_matrix.png
│   │   │   └── embedding_data.csv
│   │   ├── results_rf/
│   │   │   ├── metrics.txt
│   │   │   ├── confusion_matrix.png
│   │   │   └── embedding_data.csv
│   │   └── results_svm_rbf/
│   │       ├── metrics.txt
│   │       ├── confusion_matrix.png
│   │       └── embedding_data.csv
│   ├── 2/
│   └── 3/
├── New_results/                           # Training results (NEW dataset)
├── extract_features_pipeline.py           # 6-mer feature extraction
├── train_lr_pipeline.py                   # Logistic Regression training
├── train_rf_pipeline.py                   # Random Forest training
├── train_svm_rbf_pipeline.py              # SVM-RBF training
├── plot_pca_tsne.py                       # PCA/t-SNE visualization
├── plot_all_tsne.py                       # Batch t-SNE plot generator
├── count_sequences_by_species.py          # Dataset statistics
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

**Note**: Original FASTA files are not included in this repository. Feature matrices were pre-extracted and stored as NumPy arrays.

---

## Methodology Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SPECIES CLASSIFICATION PIPELINE                     │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │   Raw FASTA  │───▶│  FASTA Parse │───▶│ N-Filtering  │
  │   Input      │    │  per species │    │ (≤10% N)     │
  └──────────────┘    └──────────────┘    └──────────────┘
                                                │
                                                ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │   4096-Dim   │───▶│  Train/Val    │───▶│   ML Model   │
  │   6-mer Vec  │    │  Split        │    │   Training   │
  └──────────────┘    └──────────────┘    └──────────────┘
        │                                        │
        ▼                                        ▼
  ┌──────────────┐                      ┌──────────────┐
  │  PCA/t-SNE   │                      │  Evaluation  │
  │  Visualiz.   │                      │  (CM, F1)    │
  └──────────────┘                      └──────────────┘
```

---

## Alignment-Free 6-mer Feature Extraction

### Feature Engineering Process

The pipeline extracts **6-mer (k-mer) frequency features** from DNA sequences:

| Parameter | Value |
|-----------|-------|
| k-mer size | 6 |
| Total features | 4⁶ = 4096 |
| Valid bases | A, C, G, T |
| N tolerance | ≤10% per sequence |
| Window step | 1 bp (overlapping) |
| Normalization | count / total_kmers |
| Output dtype | float32 |

### Algorithm

1. **Parse FASTA**: Extract sequences from multi-line FASTA files
2. **Filter N's**: Remove sequences with >10% ambiguous bases (N)
3. **Sliding Window**: Extract all overlapping 6-mers (step = 1 bp)
4. **Validation**: Keep only kmers containing [A,C,G,T]; skip invalid
5. **Counting**: Tally frequency of each valid 6-mer
6. **Normalization**: `feature[i] = count[i] / total_count`

### Why 6-mers?

- 6-mers provide sufficient complexity (4096 features) to capture species-specific patterns
- Shorter kmers (3-mers, 4-mers) lack discriminatory power
- Longer kmers require more data to estimate frequencies reliably

---

## Machine Learning Models

### Model Hyperparameters

| Parameter | Logistic Regression | Random Forest | SVM-RBF |
|-----------|---------------------|---------------|---------|
| solver | lbfgs (default) | N/A | N/A |
| max_iter | 1000 | N/A | N/A |
| n_estimators | N/A | 100 | N/A |
| max_depth | N/A | None (unlimited) | N/A |
| kernel | N/A | N/A | rbf |
| C | 1.0 (default) | N/A | 1.0 (default) |
| gamma | N/A | N/A | scale |
| class_weight | balanced | balanced | balanced |
| random_state | 42 | 42 | 42 |

### Data Split

| Split | Ratio | Stratified | Random Seed |
|-------|-------|------------|--------------|
| Train | 70% | Yes | 42 |
| Validation | 15% | Yes | 42 |
| Test | 15% | Yes | 42 |

**Note**: This project uses holdout validation, not cross-validation.

### Class Imbalance Strategy

All models use `class_weight='balanced'` to handle imbalanced class distributions. This automatically adjusts weights inversely proportional to class frequencies.

### SVM Subsampling

For SVM with RBF kernel, datasets larger than 15,000 samples are subsampled to ensure tractable training times. Subsampling uses `random_state=42` for reproducibility.

---

## PCA and t-SNE Visualizations

### Visualization Pipeline

```
Raw Features (4096-D)
        │
        ▼
   PCA (50-D) ──▶ Retain variance for t-SNE
        │
        ▼
  t-SNE (2-D) ──▶ Scatter plot by species
```

### Parameters

| Method | Parameters |
|--------|------------|
| PCA (for t-SNE) | n_components=50, random_state=42 |
| PCA (2D plot) | n_components=2 |
| t-SNE | n_components=2, perplexity=30, n_iter=1000, random_state=42 |

---

## Confusion Matrix Analysis

Confusion matrices are generated for each model using **test set predictions** to evaluate classification performance per species. Plots are saved at 150 DPI for publication.

---

## Results Summary

### Test Set Performance (OLD Dataset)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **98.04%** | 98.07% | 98.04% | 97.99% |
| SVM-RBF | 96.62% | 96.77% | 96.62% | 96.59% |
| Logistic Regression | 95.49% | 96.20% | 95.49% | 95.72% |

### Validation Set Performance (OLD Dataset)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **97.81%** | 97.85% | 97.81% | 97.76% |
| SVM-RBF | 96.44% | 96.54% | 96.44% | 96.40% |
| Logistic Regression | 94.96% | 95.75% | 94.96% | 95.21% |

### Key Findings

1. **Random Forest achieves highest accuracy** (98.04% test) among all models
2. **SVM-RBF** performs comparably (96.62%) despite computational efficiency
3. **Logistic Regression** provides interpretable baseline (95.49%)
4. **Class-weighted training** effectively handles imbalanced species distribution
5. **6-mer features capture species-specific signatures** in 16S rRNA regions

---

## How to Run

### Installation

```bash
# Clone or navigate to project directory
cd v3-v4

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Feature Extraction

```bash
# Extract features from OLD dataset
python extract_features_pipeline.py
# Enter "old" when prompted

# Extract features from NEW dataset
python extract_features_pipeline.py
# Enter "new" when prompted
```

### Model Training

```bash
# Train Logistic Regression
python train_lr_pipeline.py
# Enter "old" or "new" when prompted

# Train Random Forest
python train_rf_pipeline.py

# Train SVM-RBF
python train_svm_rbf_pipeline.py
```

### Visualization

```bash
# Generate PCA and t-SNE plots from features
python plot_pca_tsne.py
# Select "old" or "new" when prompted

# Generate t-SNE plots from result embeddings
python plot_all_tsne.py
```

### Count Sequences

```bash
# Count sequences per species
python count_sequences_by_species.py
```

---

## Output Files

### After Feature Extraction
- `features_old/` or `features_new/`
  - `X_features.npy` — Feature matrix (N × 4096), float32
  - `y_labels.npy` — Label vector (N,), int32
  - `label_mapping.txt` — Species name mapping

### After Training
- `results_*/`
  - `metrics.txt` — Accuracy, precision, recall, F1
  - `confusion_matrix.png` — Confusion matrix plot
  - `embedding_data.csv` — PCA/t-SNE coordinates

---

## Reproducibility

All experiments use fixed random seeds for reproducibility:
- `random_state=42` for train/test split
- `random_state=42` for PCA/t-SNE
- `random_state=42` for model initialization
- `random_state=42` for SVM subsampling

Three repeated runs (1, 2, 3) are stored in `Old_results/` and `New_results/`.

---

## Limitations and Notes

1. **Holdout Validation**: This project uses single train/validation/test split rather than cross-validation. Results should be interpreted accordingly.

2. **Feature Extraction**: Original FASTA files are not included. Feature matrices were pre-extracted and stored as NumPy arrays.

3. **SVM Subsampling**: For large datasets (>15,000 samples), SVM uses subsampling which may affect comparability with other models.

4. **Weighted Metrics Only**: Per-class precision/recall are not saved in output files. Only weighted averages are reported.

5. **No Reverse Complement**: Current implementation uses forward strand only. Reverse complement handling is noted as a future improvement.

6. **No Primer Removal**: Primer sequences are not explicitly removed before feature extraction.

---

## Key Biological Findings

1. **Species Distinguishability**: Short 16S rRNA sequences can distinguish closely related *Escherichia* species with >98% accuracy using 6-mer features.

2. **Alignment-Free Success**: The k-mer approach achieves high accuracy without sequence alignment.

3. **Boundary of Resolution**: The ~2% misclassification rate suggests some species pairs share highly similar signatures, possibly due to horizontal gene transfer or recent divergence.

4. **Model Selection**: Random Forest's ensemble approach handles the high-dimensional 4096-feature space most effectively.

---

## Repository Notes
- `Old_results/` contains the final results reported in the paper.
- `New_results/` contains exploratory testing on an additional external dataset source used only to verify whether the pipeline generalized correctly. These outputs are not part of the final manuscript.
- Large generated feature files such as `X_features.npy` and `y_labels.npy` are excluded because of GitHub file size limitations and can be regenerated using `extract_features_pipeline.py`.
- Full FASTA datasets are also excluded for size reasons; representative sample FASTA inputs are included for demonstration and reproducibility of the workflow.
- Please contact the Author Ratna Kosuhik Appasani for obtaining the .Fasta, `X_features.npy` and `y_labels.npy` files.

---


## Future Improvements

- [ ] Integrate primer sequence removal before feature extraction
- [ ] Implement explicit duplicate sequence filtering
- [ ] Experiment with different k-mer sizes (4-mer, 8-mer)
- [ ] Add cross-validation for more robust performance estimates
- [ ] Incorporate reverse complement k-mers
- [ ] Test on additional genus-level pathogen datasets
- [ ] Add statistical significance testing between models

---

## Citation

If you use this code in your research, please cite:

```
Koushik Ratnakoushika Appasani. (2024). Species-Level Classification 
of Closely Related Bacteria Using Alignment-Free DNA Features and 
Machine Learning. GitHub Repository.
```

---

## Authors

- **Ratna Koushik Appasani** - Trinity Western University
- **Hongyuan Ding** - Trinity Western University
- **Arthur Jordan** - Trinity Western University
- **Jahin Nawar Hoque** - Trinity Western University

BIOT/BIOL/CMPT 470 -- Introduction to Bioinformatics  
Trinity Western University, Langley, BC, Canada

---

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

See `requirements.txt` for exact versions.
