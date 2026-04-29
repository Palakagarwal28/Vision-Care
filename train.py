import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import copy
import time

def build_model(num_classes=4):
    """
    Builds an EfficientNet-B0 model for transfer learning.
    Freezes initial layers, unfreezes the last block, and replaces the final layer.
    """
    print("Loading pretrained EfficientNet-B0...")
    # Load pretrained EfficientNet model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the last few layers (e.g., features[-1] block) for fine-tuning
    print("Unfreezing 'features[-1]' for fine-tuning...")
    for param in model.features[-1].parameters():
        param.requires_grad = True
        
    # Replace the final fully connected layer for our 4 classes
    # The new layer has requires_grad=True by default
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    print(f"Replaced final classifier layer to output {num_classes} classes.")
    
    return model

def train_model(model, train_loader, val_loader, class_weights=None, num_epochs=10, device='cpu'):
    """
    Trains the model with CrossEntropyLoss and Adam optimizer.
    Tracks validation accuracy and saves the best weights.
    """
    model = model.to(device)
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Only pass parameters that require gradients (the fine-tuned ones) to the optimizer
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=1e-4)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    since = time.time()
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                # Track history only if in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model if it has the best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:4f}')
    
    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return model

# ==========================================
# Example usage:
# ==========================================
if __name__ == "__main__":
    from dataset_loader import create_dataloaders
    
    # 1. Determine the device to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Build the transfer learning model
    model = build_model(num_classes=4)
    
    # 3. Load the data (Make sure to point to your actual dataset path)
    dataset_path = "./dataset" # Replace with your dataset path
    try:
        train_loader, val_loader, classes, class_weights = create_dataloaders(
            data_dir=dataset_path, 
            batch_size=32
        )
        
        # 4. Train the model
        trained_model = train_model(
            model, 
            train_loader, 
            val_loader, 
            class_weights=class_weights,
            num_epochs=10, 
            device=device
        )
        
        # 5. Save the trained weights
        torch.save(trained_model.state_dict(), 'eye_model.pth')
        print("Model saved to 'eye_model.pth'")
        
    except FileNotFoundError:
        print(f"Dataset path '{dataset_path}' not found. Please update it before running.")
