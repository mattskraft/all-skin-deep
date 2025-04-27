"""
Neural Style Transfer Utilities
Contains helper functions and classes for neural style transfer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm

# Define device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image size
IMSIZE = 512 if torch.cuda.is_available() else 128  # use small size if no GPU

# Define image transformation pipeline
loader = transforms.Compose([
    transforms.Resize(IMSIZE),  # Resize the smaller edge to `IMSIZE` (preserves aspect ratio)
    transforms.CenterCrop(IMSIZE),  # crop to square of size IMSIZE x IMSIZE from center
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_path):
    """
    Load an image from a file path, transform it and return it as a tensor
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (image tensor, image name without extension)
    """
    image = Image.open(image_path)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float), Path(image_path).stem

# Content loss definition
class ContentLoss(nn.Module):
    """
    Content Loss module for Neural Style Transfer
    """
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# Gram matrix for style calculation
def gram_matrix(input):
    """
    Calculate the Gram Matrix of a given tensor
    
    Args:
        input: Feature maps tensor
        
    Returns:
        tensor: Gram matrix
    """
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

# Style loss definition
class StyleLoss(nn.Module):
    """
    Style Loss module for Neural Style Transfer
    """
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# Normalization module
class Normalization(nn.Module):
    """
    Normalization module for preprocessing images before feeding into VGG19
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std
    

# Function to build the style transfer model
def get_style_model_and_losses(cnn,
                               style_img,
                               content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    """
    Build the style transfer model and setup loss calculations
    
    Args:
        cnn: Pre-trained CNN model
        style_img: Style image tensor
        content_img: Content image tensor
        content_layers: Layers to use for content loss
        style_layers: Layers to use for style loss
        
    Returns:
        tuple: (model, style_losses, content_losses)
    """
    # Normalization for preprocessing
    normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    normalization_std = torch.tensor([0.229, 0.224, 0.225])
    normalization = Normalization(normalization_mean, normalization_std)

    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# Main style transfer function
def run_style_transfer(cnn,
                       content_img, style_img, input_img, num_steps=1000,
                       style_weight=1000000, content_weight=1, verbose=True,
                       pbar=None):
    """
    Run the style transfer.
    
    Args:
        cnn: Pre-trained CNN model
        content_img: Content image tensor
        style_img: Style image tensor
        input_img: Input image tensor (starting point)
        num_steps: Number of optimization steps
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        verbose: Whether to print progress
        pbar: Optional progress bar
        
    Returns:
        tensor: Stylized image tensor
    """

    if verbose:
        print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     style_img,
                                                                     content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly
    model.eval()
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img])

    if verbose:
        print('Optimizing..')
    run = [0]
    
    # Initialize progress bar if provided
    progress_bar = pbar if pbar is not None else None
    
    while run[0] <= num_steps:
        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            
            # Update progress bar if provided
            if progress_bar is not None:
                progress_bar.update(1)
            
            if (run[0] % 100 == 0) and verbose and progress_bar is None:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)
        
        # Break if we've completed all steps
        if run[0] > num_steps:
            break

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

# Function to save generated image
def save_generated_image(output_img, output_file):
    """
    Save a generated image tensor to a file.
    
    Args:
        output_img: The output image tensor (C,H,W) format or numpy array
        output_file: Full path where to save the image
        
    Returns:
        Path: Path object pointing to the saved file
    """
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert tensor to PIL Image
    if isinstance(output_img, torch.Tensor):
        # Move to CPU if on GPU
        if output_img.is_cuda:
            output_img = output_img.cpu()
        
        # Remove batch dimension if present
        if output_img.dim() == 4:
            output_img = output_img.squeeze(0)
            
        # Convert to PIL image
        to_pil = transforms.ToPILImage()
        output_img = to_pil(output_img.clamp(0, 1))  # Assuming values in [0,1]
    
    # If it's already a numpy array, convert directly
    elif isinstance(output_img, np.ndarray):
        # If values exceed 1.0, assume they're already in [0,255]
        if output_img.max() > 1.0:
            output_img = output_img.astype(np.uint8)
        else:
            output_img = (output_img * 255).astype(np.uint8)
        output_img = Image.fromarray(output_img)
    
    # Save the image
    output_img.save(output_file)
    
    return output_file

# Function to get all images in a directory and subdirectories
def get_all_images(directory, extensions={".jpg", ".jpeg", ".png"}):
    """
    Get all image files from a directory and its subdirectories.
    
    Args:
        directory: Directory to search for images
        extensions: Set of allowed file extensions
        
    Returns:
        list: List of Path objects pointing to images
    """
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(directory).rglob(f"*{ext}"))
    return image_paths

def create_output_directories(input_dir, output_dir, content_images):
    """
    Create output directory structure mirroring the input directory.
    
    Args:
        input_dir: Base input directory
        output_dir: Base output directory
        content_images: List of image paths to process
    """
    for img_path in content_images:
        # Calculate relative path from input_dir
        rel_path = img_path.relative_to(input_dir)
        # Create corresponding output directory
        output_path = Path(output_dir) / rel_path.parent
        output_path.mkdir(parents=True, exist_ok=True)

def process_image_pair(cnn, content_path, style_path, output_file, 
                       num_steps=1000, style_weight=1000000, 
                       content_weight=1, pbar=None):
    """
    Process a single content-style image pair using neural style transfer.
    
    Args:
        cnn: Pre-trained CNN model
        content_path: Path to content image
        style_path: Path to style image
        output_file: Path to save the generated image
        num_steps: Number of optimization steps
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        pbar: Optional progress bar for overall processing
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Load images
        content_img, _ = image_loader(content_path)
        style_img, _ = image_loader(style_path)
        
        # Create input image (initially a copy of content image)
        input_img = content_img.clone()
        
        # Create style transfer progress bar (nested)
        st_pbar = tqdm(
            total=num_steps, 
            desc=f"Style transferring {content_path.name}", 
            leave=False
        )
        
        # Run style transfer
        output_img = run_style_transfer(
            cnn,
            content_img,
            style_img,
            input_img, 
            num_steps=num_steps,
            style_weight=style_weight, 
            content_weight=content_weight, 
            verbose=False,
            pbar=st_pbar
        )
        
        # Save the result
        save_generated_image(output_img, output_file)
        
        st_pbar.close()
        return True
        
    except Exception as e:
        print(f"Error processing {content_path}: {e}")
        return False

def write_csv(csv_path, data):
    """
    Write data to a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        data: List of rows to write
    """
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in data:
            csv_writer.writerow(row)