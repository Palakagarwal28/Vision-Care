import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import cv2
import numpy as np
from PIL import Image

class CropRetinaBorder(object):
    """
    Detects the circular retina region, crops the center, and removes black borders.
    """
    def __call__(self, img):
        # Convert RGBA to RGB properly by pasting over a black background
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            alpha = img.convert('RGBA').split()[-1]
            bg = Image.new("RGB", img.size, (0, 0, 0))
            bg.paste(img, mask=alpha)
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            
        img_np = np.array(img)
        # Convert to grayscale to find the mask
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
            
        # Threshold to find the retina region
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            # Crop the image to the bounding box
            img_np = img_np[y:y+h, x:x+w]
            
        return Image.fromarray(img_np)

# Wrapper class to apply different transforms to train and val subsets
class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def create_dataloaders(data_dir, batch_size=32, train_split=0.8):
    """
    Creates train and validation DataLoaders for a given dataset directory.
    
    Args:
        data_dir (str): Path to the dataset directory containing class subfolders.
        batch_size (int): Batch size for the DataLoaders (default: 32).
        train_split (float): Proportion of data to use for training (default: 0.8).
        
    Returns:
        train_loader, val_loader, class_names
    """
    
    # 2. Apply data augmentation (resize, normalize, random flip, rotation)
    # Training augmentations (includes random flip and rotation)
    train_transforms = transforms.Compose([
        CropRetinaBorder(),                  # Detect circular retina, crop, remove black borders
        transforms.Resize((224, 224)),       # Resize to standard size (e.g., 224x224 for ResNet)
        transforms.RandomHorizontalFlip(),   # Randomly flip the image horizontally
        transforms.RandomRotation(15),       # Randomly rotate the image by up to 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Adjust brightness and contrast
        transforms.ToTensor(),               # Convert image to PyTorch Tensor
        transforms.Normalize(                # Normalize with ImageNet stats
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Validation augmentations (no random augmentations, only resize and normalize)
    val_transforms = transforms.Compose([
        CropRetinaBorder(),                  # Apply the same cropping to validation
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 1. Load dataset using ImageFolder (loading base dataset without transforms first)
    base_dataset = datasets.ImageFolder(root=data_dir)
    class_names = base_dataset.classes
    
    # Calculate class distribution
    class_counts = [0] * len(class_names)
    for _, target in base_dataset.samples:
        class_counts[target] += 1
        
    print(f"Class distribution: {dict(zip(class_names, class_counts))}")
    
    # Calculate class weights for imbalanced datasets
    total_samples = len(base_dataset)
    num_classes = len(class_names)
    class_weights = []
    for count in class_counts:
        weight = total_samples / (num_classes * count) if count > 0 else 0
        class_weights.append(weight)
        
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    # 3. Split into train and validation (80-20)
    train_size = int(train_split * len(base_dataset))
    val_size = len(base_dataset) - train_size
    
    # Set seed for reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        base_dataset, 
        [train_size, val_size], 
        generator=generator
    )
    
    # Apply respective transforms
    train_dataset = DatasetWrapper(train_subset, transform=train_transforms)
    val_dataset = DatasetWrapper(val_subset, transform=val_transforms)

    # 4. Create DataLoader with batch size 32
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    return train_loader, val_loader, class_names, class_weights_tensor

# ==========================================
# Example usage:
# ==========================================
if __name__ == "__main__":
    # Specify the path to your dataset folder here
    dataset_path = "./dataset" # Replace with your actual path
    
    try:
        train_loader, val_loader, classes, class_weights = create_dataloaders(
            data_dir=dataset_path, 
            batch_size=32
        )
        
        print(f"Dataset classes found: {classes}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        # Example of iterating through the dataloader
        for images, labels in train_loader:
            print(f"Batch image shape: {images.shape}")
            print(f"Batch label shape: {labels.shape}")
            break # Just print the first batch
            
    except FileNotFoundError:
        print(f"Please ensure the dataset path '{dataset_path}' exists and is structured correctly.")
