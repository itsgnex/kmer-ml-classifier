#!/usr/bin/env python3
"""
Feature extraction pipeline: converting FASTA sequences to 6-mer frequency vectors.
This is the first step - we extract alignment-free k-mer features from V3-V4 amplicons.
"""
import os
import numpy as np
from itertools import product
from collections import defaultdict

# Using 6-mers because they provide enough resolution (4096 features) to distinguish
# closely related species. 4-mers are too coarse, 8-mers need way more samples.
KMER_SIZE = 6
MAX_N_RATIO = 0.1  # Allow up to 10% ambiguous bases - stricter would lose too many reads
BASE_ORDER = ['A', 'C', 'G', 'T']  # Standard nucleotide ordering for index consistency


def get_dataset_folder():
    """Prompt user for dataset folder."""
    return input("Enter dataset folder (old or new): ").strip()


def validate_folder(folder_path):
    """Check if folder exists."""
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        exit(1)


def get_fasta_files(folder_path):
    """Get all FASTA files in the folder."""
    fasta_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.fasta') or filename.endswith('.fa'):
            fasta_files.append(filename)
    return sorted(fasta_files)


def build_kmer_index():
    """
    Build a lookup table: map each possible 6-mer to a unique index.
    This lets us count occurrences efficiently without string matching every time.
    """
    kmer_to_idx = {}
    for i, kmer in enumerate(product(BASE_ORDER, repeat=KMER_SIZE)):
        kmer_to_idx[''.join(kmer)] = i
    return kmer_to_idx


def parse_fasta_file(filepath):
    """
    Parse a FASTA file and extract all sequences.
    Handles multi-line sequences (common in genomic files) by concatenating until the next header.
    """
    sequences = []
    current_seq = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Found a new sequence header - save the previous one if it exists
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            else:
                # Sequence line - convert to uppercase for consistency
                current_seq.append(line.upper())
        
        # Don't forget the last sequence in the file
        if current_seq:
            sequences.append(''.join(current_seq))
    
    return sequences


def filter_sequences(sequences):
    """
    Filter out sequences with too many ambiguous bases (N).
    These represent low-quality reads that would distort our k-mer frequency profiles.
    Keeping sequences with up to 10% N allows us to retain useful data without noise.
    """
    filtered = []
    for seq in sequences:
        n_count = seq.count('N')
        if len(seq) > 0 and n_count / len(seq) <= MAX_N_RATIO:
            filtered.append(seq)
    return filtered


def compute_kmer_frequencies(sequence, kmer_to_idx):
    """
    Convert a single DNA sequence into a 4096-dimensional frequency vector.
    We use overlapping 6-mers (step=1) to capture local compositional patterns in V3-V4.
    """
    kmer_counts = defaultdict(int)
    seq_len = len(sequence)
    
    # Slide a 6-base window across the entire sequence
    for i in range(seq_len - KMER_SIZE + 1):
        kmer = sequence[i:i + KMER_SIZE]
        # Skip any kmer containing N - these would introduce noise
        if all(base in BASE_ORDER for base in kmer):
            kmer_counts[kmer] += 1
    
    total_kmers = sum(kmer_counts.values())
    
    # Normalize by total count so sequence length doesn't affect features
    feature_vector = np.zeros(4 ** KMER_SIZE, dtype=np.float32)
    
    if total_kmers > 0:
        for kmer, count in kmer_counts.items():
            idx = kmer_to_idx[kmer]
            feature_vector[idx] = count / total_kmers
    
    return feature_vector


def main():
    """
    Main pipeline: load FASTA files per species, extract 6-mer features, save for training.
    Each species gets its own label so we can train a multi-class classifier.
    """
    folder_name = get_dataset_folder()
    
    if folder_name not in ['old', 'new']:
        print("Error: Folder must be 'old' or 'new'")
        exit(1)
    
    folder_path = folder_name
    
    validate_folder(folder_path)
    
    fasta_files = get_fasta_files(folder_path)
    
    if not fasta_files:
        print(f"No FASTA files found in '{folder_name}'.")
        exit(1)
    
    # Pre-compute the kmer index once - used for all sequences
    kmer_to_idx = build_kmer_index()
    
    output_dir = f"features_{folder_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    all_features = []
    all_labels = []
    label_mapping = {}
    species_list = []
    
    # Process each species file separately and assign integer labels
    for species_idx, filename in enumerate(fasta_files):
        species_name = os.path.splitext(filename)[0]
        species_list.append(species_name)
        label_mapping[species_idx] = species_name
        
        filepath = os.path.join(folder_path, filename)
        print(f"Reading file: {filename}")
        
        # Parse and filter sequences from this species
        sequences = parse_fasta_file(filepath)
        print(f"  Raw sequences: {len(sequences)}")
        
        sequences = filter_sequences(sequences)
        print(f"  Filtered sequences: {len(sequences)}")
        
        # Extract 6-mer frequency vector for each sequence
        for seq in sequences:
            features = compute_kmer_frequencies(seq, kmer_to_idx)
            all_features.append(features)
            all_labels.append(species_idx)
        
        print(f"  Progress: {species_idx + 1}/{len(fasta_files)} species processed")
    
    # Convert to numpy arrays for sklearn
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    # Save features and labels - these will be loaded by the training scripts
    X_path = os.path.join(output_dir, 'X_features.npy')
    y_path = os.path.join(output_dir, 'y_labels.npy')
    
    np.save(X_path, X)
    np.save(y_path, y)
    
    # Also save the label mapping so we know which integer corresponds to which species
    mapping_path = os.path.join(output_dir, 'label_mapping.txt')
    with open(mapping_path, 'w') as f:
        for idx, species in label_mapping.items():
            f.write(f"{idx} → {species}\n")
    
    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Total sequences: {len(y)}")
    print(f"Feature shape: {X.shape}")
    print(f"Number of species: {len(species_list)}")
    print(f"Species: {species_list}")
    print(f"\nOutput saved to: {output_dir}/")


if __name__ == "__main__":
    main()
