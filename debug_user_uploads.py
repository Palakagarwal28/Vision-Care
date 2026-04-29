import os
from inference import analyze_eye_image, is_valid_retina, load_model_for_inference

device = 'cpu'
model = load_model_for_inference("eye_model.pth", num_classes=4, device=device)

upload_dir = "static/uploads"
files = [f for f in os.listdir(upload_dir) if not f.endswith("_cropped.jpg")]

for f in files[-5:]: # look at the 5 most recent files
    img_path = os.path.join(upload_dir, f)
    valid = is_valid_retina(img_path)
    res = analyze_eye_image(img_path, model=model, device=device)
    print(f"File: {f} | Valid Retina: {valid} | Prediction: {res['predicted_class']} | Conf: {res['confidence_score']}")
