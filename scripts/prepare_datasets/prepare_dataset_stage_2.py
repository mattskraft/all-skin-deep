"""
Prepare Dataset for Stage 2 Training

Prepare dataset for cross-style training by combining original and style-transferred images.
This script creates a directory structure for the combined dataset, processes each class,
and splits the images into two halves for cross-training. Each half contains a mix of
original and style-transferred images. The script also handles missing style-transferred
images and ensures that the dataset is balanced across classes.
"""

import os
import random
import shutil
import argparse
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Combine original and style-transferred datasets')
    parser.add_argument('--original_dir', type=str, required=True, help='Path to original balanced dataset')
    parser.add_argument('--st_dir', type=str, required=True, help='Path to style-transferred dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output combined dataset')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def get_image_files(directory):
    """Get all image files in a directory"""
    image_extensions = ['.jpg', '.jpeg', '.png']
    files = []
    
    for ext in image_extensions:
        files.extend(list(Path(directory).glob(f'*{ext}')))
    
    return [f.name for f in files]

def create_directory_structure(base_dir):
    """Create the directory structure for the combined dataset"""
    for half in ['first_half', 'second_half']:
        half_dir = os.path.join(base_dir, half)
        os.makedirs(half_dir, exist_ok=True)
    
    print(f"Created directory structure at {base_dir}")

def process_class(class_name, original_dir, st_dir, output_dir, val_ratio, random_seed):
    """Process a single class combining original and ST images"""
    print(f"\nProcessing class: {class_name}")
    
    # Get paths
    original_class_dir = os.path.join(original_dir, class_name)
    st_class_dir = os.path.join(st_dir, class_name)
    
    # Check if directories exist
    if not os.path.exists(original_class_dir):
        print(f"Warning: Original directory {original_class_dir} does not exist. Skipping class.")
        return
    
    if not os.path.exists(st_class_dir):
        print(f"Warning: Style-transferred directory {st_class_dir} does not exist. Skipping class.")
        return
    
    # Get files
    original_files = get_image_files(original_class_dir)
    st_files = get_image_files(st_class_dir)
    
    print(f"Found {len(original_files)} original images and {len(st_files)} style-transferred images")
    
    # Get common files (accounting for failed style transfers)
    common_files = list(set(original_files).intersection(set(st_files)))
    missing_files = list(set(original_files) - set(st_files))
    
    if missing_files:
        print(f"Note: {len(missing_files)} images missing from style-transferred dataset")
    
    # Mix shuffled files and split into two halves
    first_half, second_half = mix_and_split_files(
        common_files, missing_files, original_class_dir, st_class_dir, val_ratio, random_seed
    )
    
    # Create class directories
    for half in ['first_half', 'second_half']:
        os.makedirs(os.path.join(output_dir, half, class_name), exist_ok=True)
    
    # Copy files
    copy_files(first_half, os.path.join(output_dir, 'first_half', class_name))
    copy_files(second_half, os.path.join(output_dir, 'second_half', class_name))
        
    return {
        'class': class_name,
        'total_original': len(original_files),
        'total_st': len(st_files),
        'missing_st': len(missing_files),
        'first_half': len(first_half),
        'second_half': len(second_half)
    }

def mix_and_split_files(common_files, missing_files, original_class_dir, st_class_dir, random_seed):
    
    # Shuffle files with fixed random seed for reproducibility
    random.seed(random_seed)
    random.shuffle(common_files)
    
    # Split common files into two halves
    half_size = len(common_files) // 2
    first_half_st = common_files[:half_size]
    first_half_orig = common_files[half_size:]
    
    # Add missing files to original half
    first_half_orig.extend(missing_files[:len(missing_files)//2])
    second_half_missing = missing_files[len(missing_files)//2:]
    
    # Prepare lists for first half
    first_half_st_paths = [(os.path.join(st_class_dir, f), f) for f in first_half_st]
    first_half_orig_paths = [(os.path.join(original_class_dir, f), f) for f in first_half_orig]
    first_half = first_half_st_paths + first_half_orig_paths
    
    # Prepare second half (reverse the selection)
    second_half_st = common_files[half_size:]
    second_half_orig = common_files[:half_size]
    
    # Add remaining missing files to second half
    second_half_orig.extend(second_half_missing)
    
    second_half_st_paths = [(os.path.join(st_class_dir, f), f) for f in second_half_st]
    second_half_orig_paths = [(os.path.join(original_class_dir, f), f) for f in second_half_orig]
    second_half = second_half_st_paths + second_half_orig_paths

    print(f"First half: {len(first_half)} images ({len(first_half_st)} ST, {len(first_half_orig)} original)")
    print(f"Second half: {len(second_half)} images ({len(second_half_st)} ST, {len(second_half_orig)} original)")

    return first_half, second_half

def copy_files(file_list, destination):
    """Copy files from source to destination"""
    for src_path, filename in file_list:
        dst_path = os.path.join(destination, filename)
        shutil.copy2(src_path, dst_path)

def main(args):
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create output directory structure
    create_directory_structure(args.output_dir)
    
    # Get class folders
    class_folders = [f.name for f in Path(args.original_dir).iterdir() if f.is_dir()]
    print(f"Found {len(class_folders)} classes: {', '.join(class_folders)}")
    
    # Process each class
    stats = []
    for class_name in class_folders:
        class_stats = process_class(
            class_name, 
            args.original_dir, 
            args.st_dir, 
            args.output_dir,
            args.random_seed
        )
        if class_stats:
            stats.append(class_stats)
    
    # Print summary
    print("\n=== Dataset Combination Complete ===")
    print(f"Combined dataset saved to: {args.output_dir}")
    
    # Print statistics
    print("\nClass statistics:")
    for stat in stats:
        print(f"Class: {stat['class']}")
        print(f"  Original images: {stat['total_original']}, Style-transferred: {stat['total_st']}, Missing ST: {stat['missing_st']}")
        print(f"  First half: {stat['first_half']} total")
        print(f"  Second half: {stat['second_half']} total")

if __name__ == "__main__":
    args = parse_args()
    main(args)
