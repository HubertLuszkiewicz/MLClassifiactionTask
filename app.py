from flask import Flask, request, jsonify, render_template
import os
# import json # Removed - no longer loading class_names.json
import shutil # Added for temporary directory cleanup
import tempfile # Added for creating temporary directories

# Assume necessary imports for TensorFlow and Keras are available
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
# Import preprocess_input based on the model architecture you are using (ResNet50)
from tensorflow.keras.applications.resnet50 import preprocess_input
# If you want detailed metrics (Precision, Recall, F1, Confusion Matrix)
from sklearn.metrics import classification_report, confusion_matrix # Added for detailed metrics
import numpy as np # Added for handling array data

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
    # Initialize temporary directory variable outside the try block
    temp_dir = None

    # Initialize metrics dictionary to return
    evaluation_metrics = {"status": "processing"}

    try:
        print("--- Received POST request to /evaluate ---") # Debug print
        print("Attempting to receive test data files...") # Debug print

        # 1. Receive and Save Test Images into a Temporary Directory
        test_files = request.files.getlist('testData')
        if not test_files:
            return jsonify({"error": "No test data files received."}), 400

        # Create a temporary directory unique to this request
        temp_dir = tempfile.mkdtemp(prefix='test_eval_')
        print(f"Saving {len(test_files)} test files to temporary directory: {temp_dir}")

        for file_storage in test_files:
            # Recreate the subdirectory structure from the filename inside the temporary directory
            # Use file_storage.filename directly as it contains the relative path from the upload
            # IMPORTANT: sanitize filename if you don't fully trust the source,
            # but image_dataset_from_directory needs the path structure.
            save_path = os.path.join(temp_dir, file_storage.filename)
            # Ensure the subdirectory for this file exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Save the file
            file_storage.save(save_path)

        print("Finished saving test files.") # Debug print


        # 2. Load Model
        selected_model_path = request.form.get('modelPath')
        if not selected_model_path:
            return jsonify({"error": "No model path selected."}), 400

        # IMPORTANT SECURITY/VALIDATION: Check if the model path is safe!
        # Prevent directory traversal attacks. Ensure path is within MODEL_SAVE_DIR.
        abs_model_save_dir = os.path.abspath(MODEL_SAVE_DIR)
        abs_selected_model_path = os.path.abspath(selected_model_path)

        # Ensure the selected model path is actually a path within the designated models directory
        if not abs_selected_model_path.startswith(abs_model_save_dir):
             print(f"SECURITY ALERT: Attempted to load model outside MODEL_SAVE_DIR: {selected_model_path}")
             return jsonify({"error": "Invalid model path specified."}), 400

        if not os.path.exists(abs_selected_model_path):
            print(f"Model file not found: {selected_model_path}")
            return jsonify({"error": f"Model file not found at server path: {selected_model_path}"}), 404

        try:
            print(f"Loading model from {selected_model_path}...") # Debug print
            # Use your load_model_from_file function
            model = load_model_from_file(abs_selected_model_path) # Using your function
            if model is None:
                 # load_model_from_file prints the specific error
                 return jsonify({"error": f"Failed to load model from {selected_model_path}."}), 500

            print("Model loaded successfully.") # Debug print
        except Exception as e: # Catch any exceptions *from* load_model_from_file if it raises instead of returns None
            print(f"Unexpected error during model loading: {e}")
            return jsonify({"error": f"Unexpected error during model loading: {e}"}), 500


        # 3. Create Test Dataset from the temporary directory
        # Labels will be inferred directly from folder names in the uploaded data
        try:
            print(f"Creating test dataset from {temp_dir}...") # Debug print
            # image_dataset_from_directory infers class names from folder names
            test_ds = image_dataset_from_directory(
                temp_dir, # Point to the temporary directory where files were saved
                labels='inferred', # Inferred from subfolder names (e.g., 'person', 'motorbike')
                label_mode='categorical', # Or 'int', MUST MATCH how the model was trained!
                image_size=(224, 224), # MUST match model's expected input size (adjust if needed)
                batch_size=32, # Batch size for evaluation
                shuffle=False # Don't shuffle test data for consistent evaluation
            )

            # Check if any classes were inferred. This can fail if the temp_dir
            # doesn't contain class subdirectories with images.
            if not test_ds.class_names:
                 print("Error: No classes inferred from test data folders.") # Debug print
                 return jsonify({"error": "No valid image files found in class subfolders within the uploaded test data."}), 400

             # Get the class names *inferred from the test data folders*
            test_dataset_class_names = test_ds.class_names
            print(f"Test dataset inferred classes: {test_dataset_class_names}") # Debug print
            print(f"Inferred {len(test_dataset_class_names)} classes.") # Debug print


            # Apply the same preprocessing used during training
            # Make sure preprocess_input matches the model architecture (ResNet50)
            test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            # Optimize data loading
            test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

            print(f"Created test dataset with {len(test_ds)} batches.") # Debug print
            print(f"Total images in test dataset: {test_ds.cardinality().numpy() * test_ds.element_spec[0].shape[0]}") # Approximate count


        except Exception as e:
            print(f"Error creating test dataset from {temp_dir}: {e}")
            # This could happen if the uploaded folder structure isn't class/image or other data issues
            return jsonify({"error": f"Error creating test dataset from uploaded files. Ensure structure is class_name/image.jpg: {e}"}), 400


        # 4. Evaluate the Model and Calculate Metrics
        try:
            print("Starting core model evaluation (loss, accuracy)...") # Debug print
            # Evaluate returns loss and metrics defined in model.compile
            # results_from_evaluate will be a list (e.g., [loss, accuracy])
            # model.metrics_names is a list (e.g., ['loss', 'accuracy'])
            # Use verbose=0 to prevent a progress bar in the console logs
            results_from_evaluate = model.evaluate(test_ds, verbose=0)
            print("Model evaluation finished.") # Debug print

            # Ensure results_from_evaluate is a list
            if not isinstance(results_from_evaluate, list):
                 results_from_evaluate = [results_from_evaluate]

            # Combine metric names and results into the dictionary
            evaluation_metrics['core_metrics'] = dict(zip(model.metrics_names, results_from_evaluate))
            print("Core evaluation metrics:", evaluation_metrics['core_metrics']) # Debug print


            # --- Calculate detailed metrics (Precision, Recall, F1, Confusion Matrix) ---
            print("Calculating detailed metrics...") # Debug print

            # Get true labels (integer indices). Convert from one-hot if necessary.
            # Collect all batches and concatenate
            all_labels = []
            for images, labels in test_ds:
                all_labels.append(labels.numpy())
            true_labels_int = np.concatenate(all_labels, axis=0)

            # Convert one-hot true labels to integer indices if label_mode='categorical' was used for the dataset
            if test_ds.element_spec[1].shape[-1] > 1 and true_labels_int.ndim > 1:
                 true_labels_int = np.argmax(true_labels_int, axis=1)


            # Get predictions (raw output, e.g., probabilities from softmax)
            # Collect all batches and concatenate
            all_predictions = []
            for images, labels in test_ds: # Iterate through test_ds again to get images for predict
                 all_predictions.append(model.predict(images)) # Predict batch by batch

            # Alternative: Predict on the whole dataset directly if memory allows (less common)
            # predictions_raw = model.predict(test_ds)

            # Concatenate predictions from batches
            predictions_raw = np.concatenate(all_predictions, axis=0)


            # Convert predictions to integer indices (e.g., using argmax for softmax output)
            predicted_labels_int = np.argmax(predictions_raw, axis=1)

            # Use the class names inferred from the test dataset for metrics reporting - they match the true labels
            sorted_class_labels = test_ds.class_names # These are sorted alphabetically by default

            # Calculate Classification Report (includes precision, recall, f1-score per class and overall)
            # target_names ensures the report uses the actual class names
            # zero_division='warn' or 0 or 1 depending on desired behavior for classes with no true/predicted samples
            report = classification_report(true_labels_int, predicted_labels_int, target_names=sorted_class_labels, output_dict=True, zero_division=0)
            evaluation_metrics['classification_report'] = report
            print("Classification report calculated.") # Debug print
            print("Report:", report)


            # Calculate Confusion Matrix
            cm = confusion_matrix(true_labels_int, predicted_labels_int)
            # Convert confusion matrix (numpy array) to list for JSON serialization
            evaluation_metrics['confusion_matrix'] = cm.tolist()
            # Add the order of classes for the confusion matrix (matches report order)
            evaluation_metrics['confusion_matrix_labels'] = sorted_class_labels
            print("Confusion matrix calculated.") # Debug print


            evaluation_metrics['status'] = 'success' # Update overall status


            # 5. Return results as JSON
            print("Returning evaluation metrics as JSON.") # Debug print
            return jsonify(evaluation_metrics)

        except Exception as e:
            print(f"Error during model evaluation or metric calculation: {e}")
            return jsonify({"error": f"Error during model evaluation or metric calculation: {e}"}), 500

    except Exception as e:
        # This catches errors that might happen before the specific try blocks inside (e.g., file saving)
        print(f"An unexpected error occurred before evaluation steps: {e}") # Debug print
        return jsonify({"error": f"An unexpected server error occurred: {e}"}), 500

    finally:
        # 6. Clean up the temporary directory
        # This block runs regardless of whether an exception occurred in the try block
        if temp_dir and os.path.exists(temp_dir):
            try:
                print(f"Cleaning up temporary directory: {temp_dir}") # Debug print
                shutil.rmtree(temp_dir)
                print("Cleanup complete.") # Debug print
            except Exception as e:
                print(f"Error cleaning up temporary directory {temp_dir}: {e}")
                # Log this error, but don't block the response.

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