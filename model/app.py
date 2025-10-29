from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(title="Waste Segregation API",
              description="API for classifying waste types using ML model")

# Add CORS middleware for ESP32 Cam
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your ESP32 IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL_PATH = "best_waste_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Define waste classes (from TrashNet dataset)
WASTE_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Map classes to biodegradable or non-biodegradable
WASTE_CATEGORIES = {
    'cardboard': 'biodegradable',
    'glass': 'non-biodegradable',
    'metal': 'non-biodegradable',
    'paper': 'biodegradable',
    'plastic': 'non-biodegradable',
    'trash': 'non-biodegradable'
}

# Image size for preprocessing
IMG_SIZE = 299


@app.get("/")
async def root():
    return {"message": "Waste Segregation API", "status": "running", "waste": "cardboard"}


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image for model prediction"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to model input size
        image = image.resize((IMG_SIZE, IMG_SIZE))

        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Image preprocessing failed: {str(e)}")


@app.post("/predict")
async def predict_waste(image_bytes: bytes = Body(...)):
    """
    Predict waste type from uploaded image
    Expects raw image bytes from ESP32 Cam
    """
    try:
        # Validate that we have image data
        if not image_bytes:
            raise HTTPException(
                status_code=400, detail="No image data provided")

        # Preprocess image
        processed_image = preprocess_image(image_bytes)

        # Make prediction
        predictions = model.predict(processed_image, verbose=0)

        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = WASTE_CLASSES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[::-1][:3]
        top_3_predictions = [
            {
                "class": WASTE_CLASSES[idx],
                "confidence": float(predictions[0][idx]),
                "category": WASTE_CATEGORIES[WASTE_CLASSES[idx]]
            }
            for idx in top_3_indices
        ]

        return {
            "predicted_waste_type": predicted_class,
            "waste_category": WASTE_CATEGORIES[predicted_class],
            "confidence": confidence,
            "top_3_predictions": top_3_predictions,
            "all_probabilities": {
                class_name: float(prob)
                for class_name, prob in zip(WASTE_CLASSES, predictions[0])
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")
