<div align="center">
  
# 👁️ Vision Care
**AI-Powered Retinal Disease Classification System**

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)

*Vision Care is an advanced medical data science project designed to detect four major types of eye conditions from raw fundus images using deep learning.*

</div>

## 📖 Overview
Vision Care bridges the gap between machine learning and clinical diagnostics. By analyzing color fundus retinal scans, the system accurately detects and classifies retinal conditions. This project features a full-stack architecture, marrying a heavy-duty PyTorch medical AI backend with a modern, glassmorphic React frontend.

**Supported Classifications:**
* 🟢 **Normal** (Healthy Retina)
* 🟡 **Cataract**
* 🟠 **Diabetic Retinopathy**
* 🔴 **Glaucoma**

## ✨ Key Features

🧠 **EfficientNet-B0 Backbone**
Powered by a fine-tuned EfficientNet deep neural network optimized for medical image classification, capable of accurately distinguishing complex vascular and nerve patterns.

🔍 **Explainable AI (Grad-CAM)**
The API doesn't just give an answer—it explains it. Using Gradient-weighted Class Activation Mapping (Grad-CAM), the UI generates dynamic heatmaps highlighting the exact regions of the retina the AI analyzed to make its clinical decision.

🛡️ **Out-of-Distribution (OOD) Rejection**
Built-in heuristic color-profile analysis ensures the model rejects corrupted, black-and-white, or non-retinal images, preventing false positives and AI hallucinations.

🔄 **Test-Time Augmentation (TTA)**
To maximize diagnostic reliability, incoming images are aggressively augmented at inference time (flipped, rotated at various angles) and processed in parallel to average out predictions and eliminate rotational bias.

📄 **Automated Medical Reporting**
With a single click, users can download a generated text report containing the detected condition, AI confidence score, risk level, and medical recommendations.

## 🛠️ Technology Stack

**Backend (AI Diagnostics)**
* **Python 3**
* **PyTorch & Torchvision** (Model Architecture, Training, TTA)
* **FastAPI** (High-performance API endpoints)
* **OpenCV** (`cv2`) (Retina cropping & mask extraction)
* **Uvicorn** (ASGI Server)

**Frontend (User Interface)**
* **React 18** (UI Framework)
* **Vite** (Build Tool)
* **Lucide React** (Medical icon system)
* **Vanilla CSS** (Custom Glassmorphism and Micro-animations)

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Node.js 18+

### 1. Clone the repository
```bash
git clone https://github.com/Palakagarwal28/Vision-Care.git
cd Vision-Care
```

### 2. Run the AI Backend
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install torch torchvision fastapi uvicorn opencv-python pillow python-multipart

# Start the FastAPI server
python main.py
```
*The backend will initialize the PyTorch weights and start on `http://localhost:8000`.*

### 3. Run the Frontend
```bash
cd frontend

# Install UI dependencies
npm install

# Start the Vite development server
npm run dev
```
*Access the beautiful UI at `http://localhost:5173`.*

## 📂 Dataset Architecture
The model is trained on a rigorously structured dataset of high-resolution retinal images, augmented during the training loop with `ColorJitter`, `RandomRotation`, and class-weight balancing to counteract inherent medical dataset imbalances.
*(Note: The 1.5GB dataset and `eye_model.pth` weights are ignored in this repository via `.gitignore` to preserve storage efficiency).*

---

> [!WARNING]
> **Medical Disclaimer**
> *Vision Care is a research-oriented data science project. It is not intended for clinical use, professional medical diagnosis, or treatment. Always consult a certified ophthalmologist or medical professional for actual retinal health concerns.*
