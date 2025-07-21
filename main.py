# main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import pickle
import json

from model.plant_model import PlantDiseaseModel

# Load model-related files
MODEL_PATH = "model/disease_model.pth"
TRANSFORM_PATH = "model/inference_transform.pkl"
LABEL_ENCODER_PATH = "model/label_encoder.pkl"
CLASS_NAME_PATH = "model/class_name.json"

# Initialize model and load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 38  # As per your training
model = PlantDiseaseModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Load transform
with open(TRANSFORM_PATH, "rb") as f:
    transform = pickle.load(f)

# Load label encoder
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# Load class names
with open(CLASS_NAME_PATH, "r") as f:
    class_names = json.load(f)

# Initialize FastAPI
app = FastAPI()

# Enable CORS (allow all for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Plant disease detection API running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load and preprocess image
        image = Image.open(file.file).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_index = torch.argmax(outputs, dim=1).item()
            predicted_label = label_encoder.inverse_transform([predicted_index])[0]
            predicted_class_name = class_names.get(predicted_label, "Unknown")

        return JSONResponse(content={
            "predicted_label": predicted_label,
            "predicted_class": predicted_class_name
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
