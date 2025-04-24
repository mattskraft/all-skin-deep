"""
Neural Style Transfer for Dataset Generation
Main script for processing images using neural style transfer
"""
import torch
from torchvision.models import vgg19, VGG19_Weights
from pathlib import Path
from tqdm.notebook import tqdm
import argparse
import random

# Import utilities
from nst_utils import (
    device, image_loader, get_all_images, save_generated_image, run_style_transfer
)

# Set default device
torch.set_default_device(device)

# Normalization values for preprocessing
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

# Function to process an entire directory
def process_directory(input_dir, output_dir, style_dir, num_steps=1000, 
                     style_weight=1000000, content_weight=1):
    """
    Process all images in a directory and its subdirectories.
    
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
    
    # Get all style images
    style_images = get_all_images(style_dir)
    if not style_images:
        raise FileNotFoundError(f"No style images found in {style_dir}")
    
    # Get all content images
    content_images = get_all_images(input_dir)
    if not content_images:
        raise FileNotFoundError(f"No content images found in {input_dir}")
    
    print(f"Found {len(content_images)} content images and {len(style_images)} style images")
    
    # Create output directory structure mirroring input
    for img_path in content_images:
        # Calculate relative path from input_dir
        rel_path = img_path.relative_to(input_dir)
        # Create corresponding output directory
        output_path = Path(output_dir) / rel_path.parent
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each content image with a randomly selected style image
    with tqdm(total=len(content_images), desc=f"Processing {input_dir.name}") as pbar:
        for content_path in content_images:
            # Select random style image
            style_path = random.choice(style_images)
            
            # Load images
            content_img, content_name = image_loader(content_path)
            style_img, style_name = image_loader(style_path)
            
            # Create input image (initially a copy of content image)
            input_img = content_img.clone()
            
            # Create style transfer progress bar (nested)
            st_pbar = tqdm(
                total=num_steps, 
                desc=f"Style transferring {content_path.name}", 
                leave=False
            )
            
            # Run style transfer
            try:
                output_img = run_style_transfer(
                    cnn, cnn_normalization_mean, cnn_normalization_std,
                    content_img, style_img, input_img, 
                    num_steps=num_steps,
                    style_weight=style_weight, 
                    content_weight=content_weight, 
                    verbose=False,
                    pbar=st_pbar
                )
                
                # Calculate output path
                rel_path = content_path.relative_to(input_dir)
                output_file = Path(output_dir) / rel_path
                
                # Save the result
                save_generated_image(output_img, output_file)
                
            except Exception as e:
                print(f"Error processing {content_path}: {e}")
            
            finally:
                st_pbar.close()
                pbar.update(1)

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
    print(f"Using device: {device}")
    
    # Create output directory
    content_path = Path(args.content)
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Process the entire content directory recursively
    print(f"Processing images in {content_path}...")
    process_directory(
        content_path, 
        output_base,
        args.style,
        num_steps=args.steps,
        style_weight=args.style_weight, 
        content_weight=args.content_weight
    )
    
    print("Style transfer complete!")

if __name__ == "__main__":
    main()
