import os
import random
from inference import load_model_for_inference, predict_single_image

device = 'cpu'
model = load_model_for_inference("eye_model.pth", num_classes=4, device=device)

classes = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
dataset_dir = "dataset"

print("--- RUNNING INFERENCE TEST ---")
for c in classes:
    class_dir = os.path.join(dataset_dir, c)
    if not os.path.exists(class_dir):
        print(f"Skipping {class_dir}")
        continue
    images = os.listdir(class_dir)
    # pick 2 random images
    for img_name in random.sample(images, min(2, len(images))):
        img_path = os.path.join(class_dir, img_name)
        
        pred, conf = predict_single_image(img_path, model, classes, device)
        print(f"True: {c} | Pred: {pred} | Conf: {conf}")
