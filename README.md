🧠 Model Architecture
Developed using PyTorch with a custom CNN.

Trained on 38 plant disease categories.

Achieved ~99.5% validation accuracy by epoch 25.

Contains 5 convolutional blocks followed by fully connected layers and dropout for regularization.

🚀 Tech Stack
Backend: FastAPI

Frontend: HTML + Jinja2

Model Serving: PyTorch

Templating: Jinja2

File Handling: shutil, uuid, os

🌱 How It Works
User uploads a plant leaf image.

The image is saved to the /static/uploads/ folder.

The FastAPI backend loads the image, applies necessary transformations, and performs prediction using the CNN model.

The prediction is returned and displayed on the frontend.

📦 Dependencies
Install required packages:

bash
Copy
Edit
pip install fastapi uvicorn jinja2 torch torchvision scikit-learn numpy pillow
🛠️ Running the App
Navigate to project root:

bash
Copy
Edit
cd path/to/project_root
Start the FastAPI server:

bash
Copy
Edit
uvicorn main:app --reload
Visit in browser:

cpp
Copy
Edit
http://127.0.0.1:8000
📷 Sample Prediction Flow
Upload a leaf image via the UI.

The backend saves and transforms the image.

The trained model returns the predicted class name (from 38 classes).

UI displays result and uploaded image.

⚙️ Notes
Ensure the folder structure is correct, especially model/ and static/uploads/.

The model was trained with num_classes=38. Double-check this if modifying the architecture.

The model and preprocessing pipeline must be consistent (i.e., same image size and normalization as used during training).

📁 Key Files Explained
plant_model.py: Defines the CNN model used for classification.

main.py: FastAPI routes for uploading and predicting.

class_names.json: Stores the mapping of indices to plant disease names.

inference_transform.pkl: Contains preprocessing steps (resize, normalize).

label_encoder.pkl: Translates predicted label indices to human-readable class names.

🧪 Future Enhancements
Add real-time webcam inference using YOLO for live plant detection.

Provide disease-specific recommendations.

Deploy on Render, AWS, or Hugging Face Spaces.

