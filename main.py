from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import uuid
import torch

from inference import analyze_eye_image, load_model_for_inference
from grad_cam import generate_gradcam_overlay

app = FastAPI(title="Vision Care API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directories exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/heatmaps", exist_ok=True)

# Mount static files so images can be accessed via URL (e.g., /static/heatmaps/image.jpg)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model globally to avoid reloading on each request
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
try:
    print("Loading model for FastAPI server...")
    # Initialize the model and target layer for Grad-CAM
    global_model = load_model_for_inference("eye_model.pth", num_classes=4, device=device)
    target_layer = global_model.features[-1] # Target layer for EfficientNet-B0
    MODEL_LOADED = True
except Exception as e:
    print(f"Warning: Could not load model on startup. Ensure 'eye_model.pth' exists. Error: {e}")
    MODEL_LOADED = False
    global_model = None
    target_layer = None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODEL_LOADED:
        return JSONResponse(
            status_code=500, 
            content={"error": "Model not loaded. Please train the model and generate 'eye_model.pth' first."}
        )
        
    try:
        # 1. Save uploaded image with a unique ID
        file_ext = file.filename.split(".")[-1]
        unique_id = str(uuid.uuid4())
        upload_path = f"static/uploads/{unique_id}.{file_ext}"
        
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1.5 Save a cropped version for debugging
        from dataset_loader import CropRetinaBorder
        from PIL import Image
        cropper = CropRetinaBorder()
        debug_img = Image.open(upload_path).convert('RGBA')
        
        # Manually paste over black if transparent to mimic dataset_loader
        bg = Image.new("RGB", debug_img.size, (0, 0, 0))
        bg.paste(debug_img, mask=debug_img.split()[3])
        
        debug_cropped = cropper(bg)
        debug_cropped_path = f"static/uploads/{unique_id}_cropped.jpg"
        debug_cropped.save(debug_cropped_path)
            
        # 2. Run Inference using the pre-loaded global model
        analysis = analyze_eye_image(upload_path, model=global_model, device=device)
        
        # 3. Generate Grad-CAM Heatmap
        heatmap_path = f"static/heatmaps/{unique_id}_heatmap.jpg"
        
        # Map predicted class name to its index for Grad-CAM target
        classes = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
        class_idx = classes.index(analysis['predicted_class'])
        
        generate_gradcam_overlay(
            image_path=upload_path,
            model=global_model,
            target_layer=target_layer,
            output_path=heatmap_path,
            device=device,
            class_idx=class_idx
        )
        
        # 4. Construct API Response
        return {
            "prediction": analysis['predicted_class'],
            "confidence": analysis['confidence_score'],
            "risk_level": analysis['risk_level'],
            "recommendation": analysis['suggested_action'],
            "heatmap_url": f"/{heatmap_path}"
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    print("Starting Vision Care API Server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
