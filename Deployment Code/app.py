import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = os.environ.get('MODEL_PATH', 'saved_models/qr_code_authentication_model_new.keras')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = img.reshape(1, 224, 224, 1)
    return img

def predict_qr_code(image_path):
    processed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(processed_image)
    probability = prediction[0][0]
    class_predicted = 'second_print' if probability > 0.5 else 'first_print'
    return probability, class_predicted

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            probability, predicted_class = predict_qr_code(file_path)
            os.remove(file_path)  # Clean up
            return jsonify({
                'predicted_class': predicted_class,
                'probability': float(probability)
            })
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/')
def home():
    return "Welcome to the QR Code Prediction API. Use /predict endpoint to upload an image."

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 6969))  # Default to 5000 locally
    app.run(host='0.0.0.0', port=port, debug=True)