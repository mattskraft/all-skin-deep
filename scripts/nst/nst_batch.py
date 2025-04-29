"""
Neural Style Transfer for Dataset Generation

This script does batch processing of images using neural style transfer. 
"""
import torch
from torchvision.models import vgg19, VGG19_Weights
from pathlib import Path
from tqdm import tqdm
import argparse
import random
import nst_utils as utils

# Set default device
torch.set_default_device(utils.device)

def process_directory(input_dir, output_dir, style_dir, num_steps=1000, 
                     style_weight=1000000, content_weight=1):
    """
    Process all images in a directory and its subdirectories.
    Saves a CSV file mapping each generated image to its style source.
    
    Args:
        input_dir: Directory with content images
        output_dir: Directory for output images
        style_dir: Directory with style images
        num_steps: Number of optimization steps
        style_weight: Weight for style loss
        content_weight: Weight for content loss
    """
    # Load VGG19 model
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    
    # Get all style and content images
    style_images = utils.get_all_images(style_dir)
    content_images = utils.get_all_images(input_dir)
    
    # Validate image collections
    if not style_images:
        raise FileNotFoundError(f"No style images found in {style_dir}")
    if not content_images:
        raise FileNotFoundError(f"No content images found in {input_dir}")
    
    print(f"Found {len(content_images)} content images and {len(style_images)} style images")
    
    # Create output directory structure mirroring input
    utils.create_output_directories(input_dir, output_dir, content_images)
    
    # Prepare CSV file for tracking style-content pairs
    csv_path = Path(output_dir) / "style_content_mapping.csv"
    csv_data = [['generated_image', 'content_image', 'style_image']]  # Header row
    
    # Process each content image with a randomly selected style image
    with tqdm(total=len(content_images), desc=f"Processing {input_dir.name}") as pbar:
        for content_path in content_images:
            # Select random style image
            style_path = random.choice(style_images)
            
            # Get output path
            rel_path = content_path.relative_to(input_dir)
            output_file = Path(output_dir) / rel_path
            
            # Process individual image pair
            success = utils.process_image_pair(
                cnn, 
                content_path, 
                style_path, 
                output_file,
                num_steps=num_steps,
                style_weight=style_weight,
                content_weight=content_weight,
                pbar=pbar
            )
            
            # Record successful transfers
            if success:
                csv_data.append([
                    str(output_file), 
                    str(content_path),
                    str(style_path)
                ])
                
            pbar.update(1)
    
    # Write data to CSV
    utils.write_csv(csv_path, csv_data)
    print(f"Style-content mapping saved to {csv_path}")

def main():
    """Main function to run the style transfer"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Neural Style Transfer for Dataset Generation')
    parser.add_argument('--content', type=str, required=True,
                        help='Path to content images directory')
    parser.add_argument('--style', type=str, required=True,
                        help='Path to style images directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of optimization steps (default: 1000)')
    parser.add_argument('--style-weight', type=float, default=1000000,
                        help='Style weight (default: 1000000)')
    parser.add_argument('--content-weight', type=float, default=1,
                        help='Content weight (default: 1)')
    args = parser.parse_args()
    
    # Check for GPU
    print(f"Using device: {utils.device}")
    
    # Handle paths
    content_path = Path(args.content)
    style_path = Path(args.style)
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Process the entire content directory recursively
    print(f"Processing images in {content_path}...")
    process_directory(
        content_path, 
        output_base,
        style_path,
        num_steps=args.steps,
        style_weight=args.style_weight, 
        content_weight=args.content_weight
    )
    
    print("Style transfer complete!")

if __name__ == "__main__":
    main()
