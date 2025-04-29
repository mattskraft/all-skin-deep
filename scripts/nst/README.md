# Neural Style Transfer for Dataset Generation

This repository contains code for applying neural style transfer to create augmented datasets. The implementation uses PyTorch and the VGG19 model to transfer artistic styles onto content images.

## Structure

The code is organized into the following files:

- `nst_utils.py`: Contains utility functions and classes for neural style transfer
- `nst_batch.py`: Main script for batch processing of images
- `nst_single.py`: Example script for processing a single image pair

## Requirements

- PyTorch (with CUDA support recommended)
- torchvision
- PIL
- numpy
- tqdm
- pathlib

## Usage

### Batch Processing

To process entire directories of images with random style assignment:

```bash
python nst_batch.py --content data/processed/dataset_balanced \
                   --style data/processed/style-images-resized \
                   --output data/processed/nst_output
                   --steps 1000 \
                   --style-weight 1000000 \
                   --content-weight 1
```

The script expects the following directory structure:
- If your content directory contains `train` and `val` subdirectories, they will be processed separately
- If not, the entire content directory will be processed

### Single Image Processing

To apply style transfer to a single image:

```bash
python nst_single.py --content /path/to/content.jpg \
                     --style /path/to/style.jpg \
                     --output output.jpg \
                     --steps 1000 \
                     --style-weight 1000000 \
                     --content-weight 1
```

## Parameters

- `--content`: Path to content image(s)
- `--style`: Path to style image(s)
- `--output`: Path to save output image(s)
- `--steps`: Number of optimization steps (default: 1000)
- `--style-weight`: Weight for style loss (default: 1000000)
- `--content-weight`: Weight for content loss (default: 1)

## How It Works

The implementation follows the approach outlined in the original paper ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576) by Gatys et al. It uses a pre-trained VGG19 network to:

1. Extract content features from specified layers
2. Extract style features (via Gram matrices) from specified layers
3. Iteratively optimize an image to match both content and style features

## Performance Notes

- Using a GPU is highly recommended as the optimization process is computationally intensive
- The image size is automatically adjusted based on available hardware (512px with GPU, 128px with CPU)
