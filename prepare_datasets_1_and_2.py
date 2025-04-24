import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path
import argparse

'''
python prepare_datasets_1.py \
  --input_dir ~/projects/all-skin-deep/data/raw/HAM10000/images \
  --metadata ~/projects/all-skin-deep/data/raw/HAM10000/HAM10000_metadata_clean.csv \
  --output_dir ~/projects/all-skin-deep/data/ \
  --random_state 42
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare HAM10000 datasets for training')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to original HAM10000 images')
    parser.add_argument('--metadata', type=str, required=True, help='Path to HAM10000_metadata_clean.csv')
    parser.add_argument('--output_dir', type=str, required=True, help='Base output directory')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def create_directory_structure(base_dir, mode='class_splits', classes=None):
    """
    Create directory structure based on mode
    
    Parameters:
    - base_dir: Base directory to create structure in
    - mode: 'class_splits' for train/val/test with class subfolders,
            'classes_only' for just class folders,
            'test_only' for test folder with class subfolders
    - classes: List of class names
    """
    os.makedirs(base_dir, exist_ok=True)
    
    if mode == 'class_splits':
        for split in ['train', 'val']:
            for cls in classes:
                os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)
    elif mode == 'classes_only':
        for cls in classes:
            os.makedirs(os.path.join(base_dir, cls), exist_ok=True)
    elif mode == 'test_only':
        for cls in classes:
            os.makedirs(os.path.join(base_dir, 'test', cls), exist_ok=True)
    
    print(f"Created directory structure at {base_dir}")

def copy_files(df, input_dir, output_dir, mode='to_class_split', split=None):
    """
    Copy files from source to destination based on dataframe
    
    Parameters:
    - df: DataFrame with metadata
    - input_dir: Source directory
    - output_dir: Destination directory
    - mode: 'to_class_split' for copying to train/val/test folders,
            'to_class_only' for copying to class folders directly
    - split: Which split to use ('train', 'val', 'test') if mode is 'to_class_split'
    """
    copied_count = 0
    for idx, row in df.iterrows():
        filename = row['filename']
        class_name = row['dx']
        
        src_path = os.path.join(input_dir, filename)
        
        if mode == 'to_class_split':
            dst_path = os.path.join(output_dir, split, class_name, filename)
        elif mode == 'to_class_only':
            dst_path = os.path.join(output_dir, class_name, filename)
        
        shutil.copy2(src_path, dst_path)
        copied_count += 1
        
        # Print progress every 500 files
        if copied_count % 500 == 0:
            print(f"Copied {copied_count} files...")
    
    print(f"Finished copying {copied_count} files")
    return copied_count

def print_class_distribution(df, title):
    """Print class distribution in a dataframe"""
    print(f"\n{title}:")
    class_dist = df['dx'].value_counts().sort_index()
    for cls, count in class_dist.items():
        print(f"  {cls}: {count} images")
    print(f"  Total: {len(df)} images")

def main(args):
    # Load metadata
    print(f"Loading metadata from {args.metadata}")
    df = pd.read_csv(args.metadata)
    print(f"Loaded {len(df)} image metadata entries")
    
    # Get all unique classes
    classes = sorted(df['dx'].unique())
    print(f"Found {len(classes)} classes: {', '.join(classes)}")
    
    # Create the dataset directories
    test_dir = os.path.join(args.output_dir, "test_dataset")
    balanced_dir = os.path.join(args.output_dir, "balanced_dataset")
    unbalanced_dir = os.path.join(args.output_dir, "unbalanced_dataset")
    
    # Step 1: Create initial test split (15% of full dataset)
    print("\nStep 1: Creating initial 85/15 train-test split...")
    train_val_df, test_df = train_test_split(
        df, 
        test_size=0.15, 
        stratify=df['dx'], 
        random_state=args.random_state
    )
    
    # Create directory structure for test dataset
    create_directory_structure(test_dir, mode='test_only', classes=classes)
    
    # Copy test files to test directory
    copy_files(test_df, args.input_dir, test_dir, mode='to_class_split', split='test')
    
    # Save test metadata
    test_df.to_csv(os.path.join(test_dir, 'test_metadata.csv'), index=False)
    print_class_distribution(test_df, "Test split class distribution")
    
    # Step 2: Create unbalanced dataset from the remaining 85% with 70-15 split (original proportions)
    print("\nStep 2: Creating unbalanced dataset with 70/15 split...")
    
    # Split the remaining 85% into train (70% of original) and val (15% of original)
    # This means we want an approximately 82.35/17.65 split of the train_val data
    # (because 70/85 ≈ 82.35% and 15/85 ≈ 17.65%)
    unbalanced_train_df, unbalanced_val_df = train_test_split(
        train_val_df, 
        test_size=0.1765, 
        stratify=train_val_df['dx'], 
        random_state=args.random_state
    )
    
    # Create directory structure for unbalanced dataset
    create_directory_structure(unbalanced_dir, mode='class_splits', classes=classes)
    
    # Copy files to unbalanced dataset directory
    copy_files(unbalanced_train_df, args.input_dir, unbalanced_dir, mode='to_class_split', split='train')
    copy_files(unbalanced_val_df, args.input_dir, unbalanced_dir, mode='to_class_split', split='val')
    
    # Save unbalanced metadata
    unbalanced_train_df.to_csv(os.path.join(unbalanced_dir, 'train_metadata.csv'), index=False)
    unbalanced_val_df.to_csv(os.path.join(unbalanced_dir, 'val_metadata.csv'), index=False)
    
    print_class_distribution(unbalanced_train_df, "Unbalanced train split class distribution")
    print_class_distribution(unbalanced_val_df, "Unbalanced validation split class distribution")
    
    # Step 3: Create balanced dataset from the remaining 85% (no train/val split)
    print("\nStep 3: Creating balanced dataset from the remaining 85% (no splits)...")
    
    # Find the smallest class and its count
    class_counts = train_val_df['dx'].value_counts()
    smallest_class = class_counts.index[-1]
    smallest_count = class_counts.min()
    
    print(f"Smallest class: {smallest_class} with {smallest_count} samples")
    
    # Select all samples from smallest class
    balanced_df = train_val_df[train_val_df['dx'] == smallest_class].copy()
    
    # Randomly sample from other classes to match smallest class count
    for class_name in classes:
        if class_name != smallest_class:
            class_df = train_val_df[train_val_df['dx'] == class_name]
            sampled_df = class_df.sample(n=smallest_count, random_state=args.random_state)
            balanced_df = pd.concat([balanced_df, sampled_df])
    
    # Create directory structure for balanced dataset (just class folders, no splits)
    create_directory_structure(balanced_dir, mode='classes_only', classes=classes)
    
    # Copy files to balanced dataset directory
    copy_files(balanced_df, args.input_dir, balanced_dir, mode='to_class_only')
    
    # Save balanced metadata
    balanced_df.to_csv(os.path.join(balanced_dir, 'balanced_metadata.csv'), index=False)
    
    print_class_distribution(balanced_df, "Balanced dataset class distribution")
    
    # Final summary
    print("\n=== Dataset Preparation Complete ===")
    print(f"Test dataset: {test_dir}")
    print(f"  Test: {len(test_df)} images")
    
    print(f"\nUnbalanced dataset: {unbalanced_dir}")
    print(f"  Train: {len(unbalanced_train_df)} images")
    print(f"  Validation: {len(unbalanced_val_df)} images")
    
    print(f"\nBalanced dataset: {balanced_dir}")
    print(f"  Total: {len(balanced_df)} images (no train/val split)")
    print(f"  Per class: {smallest_count} images")

if __name__ == "__main__":
    args = parse_args()
    main(args)
