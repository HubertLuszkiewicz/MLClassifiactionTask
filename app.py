from flask import Flask, request, jsonify, render_template
import os
from functions.resnet import save_resnet50, load_resnet50
import tensorflow as tf
import numpy as np

# Import the ResNet50 model class
from tensorflow.keras.applications import ResNet50
# Import the utility functions specific to ResNet50 for preprocessing and decoding predictions
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.resnet50 import preprocess_input

save_resnet50()
RESNET_MODEL_FILENAME = "resnet.h5"

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
    model = load_resnet50(selected_model_path)

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