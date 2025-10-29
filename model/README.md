# Waste Segregation API

A FastAPI server that uses a machine learning model to classify waste types from images captured by ESP32 Cam.

## Features

- Classifies waste into 6 categories: cardboard, glass, metal, paper, plastic, trash
- Accepts image uploads via HTTP POST
- Returns JSON response with predicted waste type and confidence scores
- CORS enabled for ESP32 Cam integration

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
# or
pip install -e .
```

## Usage

### Running the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### GET /

Returns server status.

#### POST /predict

Upload an image to get waste classification.

**Request:**

- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**

```json
{
  "predicted_waste_type": "plastic",
  "confidence": 0.9876,
  "top_3_predictions": [
    { "class": "plastic", "confidence": 0.9876 },
    { "class": "paper", "confidence": 0.0089 },
    { "class": "cardboard", "confidence": 0.0035 }
  ],
  "all_probabilities": {
    "cardboard": 0.0,
    "glass": 0.0,
    "metal": 0.0,
    "paper": 0.0089,
    "plastic": 0.9876,
    "trash": 0.0
  }
}
```

### ESP32 Cam Integration

The ESP32 Cam can send images to this API using HTTP POST requests with multipart/form-data.

Example ESP32 code:

```cpp
// Capture image and send to API
// (Implementation depends on your ESP32 Cam library)
```

## Model Details

- **Model**: TensorFlow/Keras model trained on TrashNet dataset
- **Input Size**: 299x299 pixels
- **Classes**: cardboard, glass, metal, paper, plastic, trash
- **Accuracy**: ~95% on validation set

## Development

The model was trained using transfer learning with MobileNetV2 base and fine-tuned for waste classification.
