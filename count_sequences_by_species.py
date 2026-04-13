#!/usr/bin/env python3
"""
Simple utility script to count sequences per species in our FASTA files.
Useful for checking class distribution before training - helps us understand
the severe imbalance (e.g., 15k+ E. coli vs 360 E. ruysiae).
"""
import os


def get_folder_name():
    """Prompt user for folder name."""
    return input("Enter folder name (e.g., old or new): ").strip()


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


def count_sequences(file_path):
    """
    Count sequences in a FASTA file.
    Each sequence starts with '>' so we just count headers.
    """
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count


def get_species_name(filename):
    """Extract species name from filename (without extension)."""
    return os.path.splitext(filename)[0]


def main():
    """Count and report sequences per species."""
    folder_name = get_folder_name()
    folder_path = folder_name
    
    validate_folder(folder_path)
    
    fasta_files = get_fasta_files(folder_path)
    
    if not fasta_files:
        print(f"No FASTA files found in '{folder_name}'.")
        exit(1)
    
    species_counts = {}
    
    # Loop through each species file and count
    for filename in fasta_files:
        file_path = os.path.join(folder_path, filename)
        print(f"Reading file: {filename}")
        
        count = count_sequences(file_path)
        print(f"Count: {count} sequences")
        
        species_name = get_species_name(filename)
        species_counts[species_name] = count
    
    print("\n## Species Counts")
    
    for species in sorted(species_counts.keys()):
        print(f"{species}: {species_counts[species]}")
    
    print("-" * 10)
    
    total_sequences = sum(species_counts.values())
    print(f"\nTotal number of sequences across all species: {total_sequences}")


if __name__ == "__main__":
    main()
