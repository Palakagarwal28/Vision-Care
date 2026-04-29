import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from dataset_loader import CropRetinaBorder

class GradCAM:
    """
    Calculates Grad-CAM heatmap for a specific target layer of a CNN.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to extract feature maps and gradients
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] contains the gradients with respect to the output of the layer
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        """
        Generates the heatmap for the given input tensor.
        Args:
            x: input image tensor of shape (1, C, H, W)
            class_idx: Target class index. If None, uses the class with the highest predicted score.
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        # We target the specific output class neuron
        target = output[0, class_idx]
        target.backward()

        # Get the gradients and activations saved from the hooks
        gradients = self.gradients[0]
        activations = self.activations[0]

        # Global average pooling on gradients to get the weights for each channel
        weights = torch.mean(gradients, dim=(1, 2))

        # Weight the activation channels by the corresponding gradient weights
        for i, w in enumerate(weights):
            activations[i, :, :] *= w

        # Create the heatmap by summing across channels
        heatmap = torch.sum(activations, dim=0).squeeze()
        
        # Apply ReLU to keep only positive influences on the class prediction
        heatmap = F.relu(heatmap)
        
        # Normalize the heatmap between 0 and 1
        if torch.max(heatmap) != 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.cpu().numpy()

def generate_gradcam_overlay(image_path, model, target_layer, output_path="gradcam_output.jpg", device='cpu', class_idx=None):
    """
    Generates a Grad-CAM heatmap, overlays it on the original image, and saves it.
    """
    # 1. Prepare original image and crop it so the heatmap aligns correctly
    original_image = Image.open(image_path).convert('RGB')
    cropper = CropRetinaBorder()
    cropped_image = cropper(original_image)
    
    # 2. Prepare tensor for the model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(cropped_image).unsqueeze(0).to(device)
    
    # 3. Initialize GradCAM
    grad_cam = GradCAM(model, target_layer)
    
    # 4. Generate heatmap (1D array of [0, 1])
    heatmap = grad_cam(input_tensor, class_idx=class_idx)
    
    # 5. Resize heatmap to match the cropped image size
    img_np = np.array(cropped_image)
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    
    # 6. Apply colormap to the heatmap (turns it into RGB)
    heatmap_color = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) # OpenCV uses BGR, PIL uses RGB
    
    # 7. Overlay the heatmap on the original image
    alpha = 0.5 # Transparency of the heatmap overlay
    superimposed_img = heatmap_color * alpha + img_np * (1.0 - alpha)
    superimposed_img = np.uint8(superimposed_img)
    
    # 8. Save the final image
    Image.fromarray(superimposed_img).save(output_path)
    print(f"Grad-CAM saved successfully to: {output_path}")
    return output_path

# ==========================================
# Example usage:
# ==========================================
if __name__ == "__main__":
    from inference import load_model_for_inference
    
    # Select your device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load the model we trained previously
        model = load_model_for_inference('eye_model.pth', num_classes=4, device=device)
        
        # ---------------------------------------------------------
        # SELECTING TARGET LAYER FOR COMPATIBILITY
        # ---------------------------------------------------------
        # For EfficientNet (like our model): The final feature block
        target_layer = model.features[-1]
        # ---------------------------------------------------------
        
        # Test the implementation on an image
        # test_image = 'path/to/test/image.jpg'
        # generate_gradcam_overlay(test_image, model, target_layer, output_path='gradcam_result.jpg', device=device)
        
    except FileNotFoundError:
        print("Please ensure the model file and image paths are valid.")
    except Exception as e:
        print(f"Error occurred: {e}")
