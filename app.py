from flask import Flask, request, jsonify, render_template
import os
import tensorflow as tf
import numpy as np

# Import the ResNet50 model class
from tensorflow.keras.applications import ResNet50
# Import the utility functions specific to ResNet50 for preprocessing and decoding predictions
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.models import load_model

def load_model_from_file(model_path):
    """
    Loads a Keras model from the specified file path.

    Args:
        model_path (str): The path to the saved model file (.h5, .keras)
                          or the directory (SavedModel format).

    Returns:
        tf.keras.Model or None: The loaded model object, or None if loading failed.
    """
    print(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path): # os needs to be imported
        print(f"Error: Model file/directory not found at {model_path}")
        return None

    try:
        # Use tf.keras.models.load_model
        # compile=False is often useful if you only need inference/evaluation
        # and don't need the original optimizer or loss function setup.
        # custom_objects is needed if your model architecture includes custom layers
        # that are not built-in Keras layers.
        # model = load_model(model_path, compile=False, custom_objects={'YourCustomLayer': YourCustomLayer})

        # For a standard ResNet50 without custom layers, compile=False is sufficient
        model = load_model(model_path, compile=False)

        print("Model loaded successfully.")
        return model

    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        # This could happen due to file corruption, incompatibility, missing custom objects, etc.
        return None
    

app = Flask(__name__)
MODEL_SAVE_DIR = 'trained_models'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=["GET"])
def show_training_page():
    return render_template('train_model.html')

@app.route('/train', methods=["POST"])
def train_model():
    training_files = request.files.getlist('trainingData')

    if not training_files:
        return jsonify({"error": "No training data files received."}), 400

    # training logic

    return jsonify({
        "status": "Data received",
        "message": f"Successfully received {len(training_files)} training files. Processing...",
        "training_file_count": len(training_files)
    })

@app.route('/evaluate', methods=["GET"])
def show_evaluation_page():
    return render_template('evaluate_model.html')

@app.route('/evaluate', methods=["POST"])
def evaluate_model():
    # Load test images and preprocess them
    TEST_DIR = "uploaded_test_data"
    test_files = request.files.getlist('testData')
    if not test_files:
        return jsonify({"error": "No test data files received."}), 400
    for file_storage in test_files:
        save_path = os.path.join(TEST_DIR, file_storage.filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file_storage.save(save_path)

    # Load model
    selected_model_path = request.form.get('modelPath')
    model = load_model_from_file(selected_model_path)

    test_ds = image_dataset_from_directory(
        TEST_DIR, # Point to the temporary directory where files were saved
        labels='inferred', # Inferred from subfolder names (class_a, class_b)
        label_mode='categorical', # Or 'int', depending on how your model was trained
        image_size=(224, 224), # Match your training image size
        batch_size=32,
        shuffle=False # Don't shuffle test data
    )

    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    results = model.evaluate(test_ds)
    return jsonify({
        "results": results
    })


@app.route('/list_models', methods=['GET'])
def list_available_models():
    models_list = []
    if os.path.exists(MODEL_SAVE_DIR):
        for item_name in os.listdir(MODEL_SAVE_DIR):
            item_path = os.path.join(MODEL_SAVE_DIR, item_name)
            
            models_list.append({"name": item_name, "path": item_path})

    return jsonify(models_list)

@app.route("/classify", methods=["GET"])
def show_classification_page():
    return render_template('classify_image.html')

@app.route("/classify", methods=["POST"])
def get_prediction():
    # Image file not in request
    if 'image' not in request.files:
        response = {"error": "You must upload an image."}
        return jsonify(response), 400
    
    file = request.files['image']

    # Image file not selected
    if file.filename == '':
        response = {"error": "No selected file."}
        return jsonify(response), 400
    
    # Prediction logic

    # Successfull prediction
    predicted_label = "simulated_prediction" # Replace with actual
    confidence = 0.99 # Replace with actual

    success_response = {
        "message": "Prediction successful!",
        "filename": file.filename,
        "predicted_label": predicted_label,
        "confidence": confidence
    }
    return jsonify(success_response), 200