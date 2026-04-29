import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from dataset_loader import CropRetinaBorder

def load_model_for_inference(model_path, num_classes=4, device='cpu'):
    """
    Initializes the EfficientNet-B0 architecture and loads the trained weights from disk.
    """
    print(f"Loading model from {model_path}...")
    
    # 1. Initialize the same architecture used during training
    # Note: We don't need pretrained=True or weights here since we'll load our own weights
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    # 2. Load our trained state dictionary
    # map_location ensures it loads correctly whether on CPU or GPU
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    # 3. Move model to the correct device and set to evaluation mode
    model = model.to(device)
    model.eval() # Important: sets dropout and batch norm to evaluation mode
    
    return model

def predict_single_image(image_path, model, class_names, device='cpu'):
    """
    Loads an image from disk, applies Test-Time Augmentation (TTA), and returns the predicted class.
    """
    # Load image and convert to RGB
    image = Image.open(image_path).convert('RGB')
    
    # Helper to create transform pipelines
    def get_transform(augmenter=None):
        ops = [CropRetinaBorder(), transforms.Resize((224, 224))]
        if augmenter:
            ops.append(augmenter)
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transforms.Compose(ops)

    # 1. Base transform
    t_base = get_transform()
    # 2. Horizontal Flip
    t_flip = get_transform(transforms.RandomHorizontalFlip(p=1.0))
    # 3. Rotate +10
    t_rot_p = get_transform(transforms.RandomRotation((10, 10)))
    # 4. Rotate -10
    t_rot_m = get_transform(transforms.RandomRotation((-10, -10)))
    
    tensors = [
        t_base(image).unsqueeze(0).to(device),
        t_flip(image).unsqueeze(0).to(device),
        t_rot_p(image).unsqueeze(0).to(device),
        t_rot_m(image).unsqueeze(0).to(device)
    ]
    
    # Forward pass with TTA (average the probabilities)
    with torch.no_grad():
        all_probs = []
        for t in tensors:
            outputs = model(t)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs)
            
        # Average the probabilities across all augmentations
        avg_probs = torch.mean(torch.stack(all_probs), dim=0)
        
        # Get the highest probability class from the averaged tensor
        confidence, predicted_idx = torch.max(avg_probs, 1)
        
    predicted_class = class_names[predicted_idx.item()]
    conf_score_pct = confidence.item() * 100.0
    
    return predicted_class, conf_score_pct

def is_valid_retina(image_path):
    """
    Validates if the uploaded image has the color profile of a retina scan.
    Retinas are predominantly red/orange. If the image is noise or greyscale, reject it.
    """
    import cv2
    import numpy as np
    img = cv2.imread(image_path)
    if img is None: return False
    
    # Calculate average color BGR
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    b, g, r = avg_color
    
    if r < g + 5 or r < b + 5:
        return False
    return True

def analyze_eye_image(image_path, model=None, model_path='eye_model.pth', device='cpu'):
    """
    Complete pipeline to load model, predict class, and return a comprehensive analysis 
    including risk level and suggested medical actions.
    """
    classes = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
    
    if not is_valid_retina(image_path):
        return {
            'predicted_class': 'Invalid / Unrecognized Scan',
            'confidence_score': '0.0%',
            'risk_level': 'Unknown',
            'suggested_action': 'The uploaded image does not appear to be a valid retinal scan. Please upload a clear, color fundus image.'
        }
        
    # Load model if not provided
    if model is None:
        model = load_model_for_inference(model_path, num_classes=4, device=device)
    
    # Get prediction and confidence
    pred_class, confidence = predict_single_image(image_path, model, classes, device)
    
    # Determine Risk Level and Action
    if confidence < 55.0:
        pred_class = "Invalid / Unrecognized Scan"
        risk_level = "Unknown"
        action = "The AI is not confident in this image. Please ensure you are uploading a clear, centered retinal scan."
    else:
        if pred_class == 'Normal':
            risk_level = 'Low'
            action = "Maintain routine annual eye exams. No immediate action required."
        elif pred_class == 'Cataract':
            risk_level = 'Medium'
            action = "Schedule a non-urgent consultation with an ophthalmologist to discuss potential surgical options if vision is impaired."
        elif pred_class == 'Diabetic Retinopathy':
            risk_level = 'High'
            action = "URGENT: Consult a retina specialist immediately. Strict blood sugar control is strongly advised."
        elif pred_class == 'Glaucoma':
            risk_level = 'High'
            action = "URGENT: See an ophthalmologist immediately for intraocular pressure testing to prevent irreversible nerve damage."
        else:
            risk_level = 'Unknown'
            action = "Consult an eye care professional for a complete diagnosis."
        
    return {
        'predicted_class': pred_class,
        'confidence_score': f"{confidence:.1f}%",
        'risk_level': risk_level,
        'suggested_action': action
    }

# ==========================================
# Example usage:
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Replace these with the exact names of your dataset subfolders in alphabetical order.
    # PyTorch ImageFolder sorts classes alphabetically by default.
    classes = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
    
    try:
        # Load the saved model
        model = load_model_for_inference('eye_model.pth', num_classes=4, device=device)
        print("Model successfully loaded!")
        
        # Example of how to predict an image:
        # test_img = 'path/to/some/test_image.jpg'
        # pred_class, conf = predict_single_image(test_img, model, classes, device)
        # print(f"Prediction: {pred_class} (Confidence: {conf:.1f}%)")
        
    except FileNotFoundError:
        print("Error: 'eye_model.pth' not found. Please run the training script first.")
