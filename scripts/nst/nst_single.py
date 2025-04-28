"""
Example script for performing Neural Style Transfer on a single image pair
"""
import torch
import argparse
from pathlib import Path
from torchvision.models import vgg19, VGG19_Weights

# Import utilities
from nst_utils import device, image_loader, save_generated_image, run_style_transfer

# Set default device
torch.set_default_device(device)

# Normalization values for preprocessing
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

def main():
    """Run style transfer on a single content-style image pair"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Neural Style Transfer on a single image')
    parser.add_argument('--content', type=str, required=True,
                        help='Path to content image')
    parser.add_argument('--style', type=str, required=True,
                        help='Path to style image')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output image')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of optimization steps (default: 1000)')
    parser.add_argument('--style-weight', type=float, default=1000000,
                        help='Style weight (default: 1000000)')
    parser.add_argument('--content-weight', type=float, default=1,
                        help='Content weight (default: 1)')
    args = parser.parse_args()
    
    # Check for GPU
    print(f"Using device: {device}")
    
    # Load VGG19 model
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    
    # Load images
    content_img, _ = image_loader(args.content)
    style_img, _ = image_loader(args.style)
    
    # Create input image (initially a copy of content image)
    input_img = content_img.clone()
    
    # Run style transfer
    print(f"Running style transfer for {args.steps} steps...")
    output = run_style_transfer(
        cnn, cnn_normalization_mean, cnn_normalization_std,
        content_img, style_img, input_img,
        num_steps=args.steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        verbose=True
    )
    
    # Save the result
    output_path = Path(args.output)
    save_generated_image(output, output_path)
    print(f"Style transfer complete! Result saved to {output_path}")

if __name__ == "__main__":
    main()
