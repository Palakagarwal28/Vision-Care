import os
import random
from inference import load_model_for_inference, predict_single_image

device = 'cpu'
model = load_model_for_inference("eye_model.pth", num_classes=4, device=device)

classes = ['cataract', 'diabetic_retinopathy', 'normal']
dataset_dir = "dataset"

print("--- FINDING HIGH CONFIDENCE TEST IMAGES ---")
for c in classes:
    class_dir = os.path.join(dataset_dir, c)
    if not os.path.exists(class_dir):
        continue
    images = os.listdir(class_dir)
    found = False
    
    # Try a few random images until we find one with >80% confidence
    for img_name in random.sample(images, min(20, len(images))):
        img_path = os.path.join(class_dir, img_name)
        pred, conf = predict_single_image(img_path, model, classes, device)
        if pred == c and conf > 80.0:
            print(f"[{c.upper()}] SUCCESS! Please upload this exact file: {img_path} (Confidence: {conf:.1f}%)")
            found = True
            break
            
    if not found:
        print(f"[{c.upper()}] Could not find a >80% confident image in 20 tries.")
