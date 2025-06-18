from flask import Flask, request, jsonify, render_template
import os
# import json # Not needed if not loading class_names.json file
import shutil # Added for temporary directory cleanup
import tempfile # Added for creating temporary directories

# Assume necessary imports for TensorFlow and Keras are available
import tensorflow as tf
from tensorflow.keras.models import load_model # To load the saved model
from tensorflow.keras.preprocessing import image_dataset_from_directory
# Import preprocess_input based on the model architecture you are using (ResNet50)
from tensorflow.keras.applications.resnet50 import preprocess_input
# If you want detailed metrics (Precision, Recall, F1, Confusion Matrix)
from sklearn.metrics import classification_report, confusion_matrix # Added for detailed metrics
import numpy as np # Added for handling array data
from tensorflow.keras.optimizers import Adam # Need an optimizer to compile
from tensorflow.keras.losses import CategoricalCrossentropy # Need a loss to compile

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
    temp_dir = None # Initialize temporary directory variable
    evaluation_metrics = {"status": "processing"}

    try:
        print("--- Received POST request to /evaluate ---") # Debug print

        # 1. Receive and Save Test Images into a Temporary Directory
        print("Attempting to receive and save test data files...") # Debug print
        try:
            test_files = request.files.getlist('testData')
            if not test_files:
                print("Error: No test data files received.") # Debug print
                return jsonify({"error": "No test data files received."}), 400

            # Create a temporary directory unique to this request
            temp_dir = tempfile.mkdtemp(prefix='test_eval_')
            print(f"Saving {len(test_files)} test files to temporary directory: {temp_dir}") # Debug print

            for file_storage in test_files:
                # file_storage.filename contains the relative path (e.g., 'mini_dataset/person/image.jpg')
                save_path = os.path.join(temp_dir, file_storage.filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                file_storage.save(save_path)

            print("Finished saving test files.") # Debug print

        except Exception as e:
            print(f"Error during file reception or saving: {e}")
            return jsonify({"error": f"Error receiving or saving test files: {e}"}), 500


        # --- MODIFICATION START ---
        # 2. Find the root directory within the temporary directory (e.g., 'mini_dataset')
        dataset_root_in_temp = None
        try:
            items_in_temp_dir = os.listdir(temp_dir)
            # Expect exactly one item, which is the folder the user uploaded
            if len(items_in_temp_dir) == 1 and os.path.isdir(os.path.join(temp_dir, items_in_temp_dir[0])):
                 dataset_root_name = items_in_temp_dir[0]
                 dataset_root_in_temp = os.path.join(temp_dir, dataset_root_name)
                 print(f"Identified dataset root folder inside temp dir: {dataset_root_in_temp}") # Debug print
            else:
                 # This happens if the user uploaded individual files or multiple items
                 print(f"Error: Unexpected structure inside temp directory. Expected a single root folder.") # Debug print
                 print(f"Contents: {items_in_temp_dir}") # Debug print
                 return jsonify({"error": "Uploaded test data has an unexpected structure. Please upload a single root folder containing class subfolders."}), 400 # Use 400 for bad client data

        except Exception as e:
            print(f"Error identifying dataset root folder in temp dir: {e}")
            return jsonify({"error": f"Server error processing uploaded folder structure: {e}"}), 500
        # --- MODIFICATION END ---


        # 3. Load Model
        print("Attempting to load model...") # Debug print
        try:
            selected_model_path = request.form.get('modelPath')
            if not selected_model_path:
                print("Error: No model path selected.") # Debug print
                return jsonify({"error": "No model path selected."}), 400

            abs_model_save_dir = os.path.abspath(MODEL_SAVE_DIR)
            abs_selected_model_path = os.path.abspath(selected_model_path)

            if not abs_selected_model_path.startswith(abs_model_save_dir):
                 print(f"SECURITY ALERT: Attempted to load model outside MODEL_SAVE_DIR: {selected_model_path}")
                 return jsonify({"error": "Invalid model path specified."}), 400

            if not os.path.exists(abs_selected_model_path):
                print(f"Model file not found: {selected_model_path}")
                return jsonify({"error": f"Model file not found at server path: {selected_model_path}"}), 404

            model = load_model_from_file(abs_selected_model_path)
            if model is None:
                 return jsonify({"error": f"Failed to load model from {selected_model_path}. Check server logs for details."}), 500

            print("Model loaded successfully.") # Debug print

        except Exception as e: # Catch any unexpected errors during model loading
             print(f"Unexpected error during model loading: {e}")
             return jsonify({"error": f"Unexpected error during model loading: {e}"}), 500


        # 4. Create Test Dataset from the identified dataset root within the temporary directory
        print("Attempting to create test dataset...") # Debug print
        try:
            # MODIFICATION: Point image_dataset_from_directory to the *subfolder*
            test_ds = image_dataset_from_directory(
                dataset_root_in_temp, # MODIFIED: Use the identified root folder path
                labels='inferred', # Inferred from subfolder names (e.g., 'person', 'motorbike' inside mini_dataset)
                label_mode='categorical', # Or 'int', MUST MATCH how the model was trained!
                image_size=(224, 224), # MUST match model's expected input size (adjust if needed)
                batch_size=32, # Batch size for evaluation
                shuffle=False # Don't shuffle test data for consistent evaluation
            )

            # Check if any classes were inferred. This can fail if dataset_root_in_temp
            # doesn't contain class subdirectories with images.
            if not test_ds.class_names:
                 print("Error: No classes inferred from test data folders.") # Debug print
                 return jsonify({"error": "No valid image files found in class subfolders within the uploaded test data. Ensure structure is class_name/image.jpg"}), 400

            test_dataset_class_names = test_ds.class_names
            print(f"Test dataset inferred classes: {test_dataset_class_names}") # Debug print
            print(f"Inferred {len(test_dataset_class_names)} classes.") # Debug print

            # --- You might want to add a check here to see if the number of inferred classes
            # matches the number of classes the model was trained on. This requires knowing
            # how many classes the loaded model expects (e.g., from loaded class_names.json
            # IF you were using it, or perhaps an attribute saved with the model).
            # If your model has 8 classes, you expect len(test_dataset_class_names) == 8.
            # if len(test_dataset_class_names) != 8: # Replace 8 with your actual number of classes
            #      print(f"Warning: Inferred {len(test_dataset_class_names)} classes, but model expects 8.")
                 # Decide if this is a fatal error or just a warning

            # Apply the same preprocessing used during training
            test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            # Optimize data loading
            test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

            print(f"Created test dataset with {len(test_ds)} batches.") # Debug print

        except Exception as e:
            print(f"Error creating test dataset from {dataset_root_in_temp}: {e}")
            return jsonify({"error": f"Error creating test dataset from uploaded files. Ensure structure is class_name/image.jpg: {e}"}), 400


        # 5. Evaluate the Model and Calculate Metrics
        print("Attempting to evaluate model and calculate metrics...") # Debug print
        try:
            print("Starting core model evaluation (loss, accuracy)...") # Debug print

            # ADD THIS COMPILATION STEP (if you used compile=False in load_model)
            try:
                model.compile(
                    optimizer=Adam(), # Optimizer is required but state not used by evaluate
                    loss=CategoricalCrossentropy(), # Or 'categorical_crossentropy' string, must match training
                    metrics=['accuracy'] # Or other metrics
                )
                print("Model compiled for evaluation.") # Debug print
            except Exception as e:
                 print(f"Error compiling model for evaluation: {e}")
                 # Decide if this is a fatal error - usually it is if evaluate() is needed
                 return jsonify({"error": f"Error compiling model for evaluation: {e}"}), 500
            # END COMPILATION STEP


            results_from_evaluate = model.evaluate(test_ds, verbose=0)
            print("Model evaluation finished.") # Debug print

            if not isinstance(results_from_evaluate, list):
                 results_from_evaluate = [results_from_evaluate]

            evaluation_metrics['core_metrics'] = dict(zip(model.metrics_names, results_from_evaluate))
            print("Core evaluation metrics:", evaluation_metrics['core_metrics'])


            # --- Calculate detailed metrics (Precision, Recall, F1, Confusion Matrix) ---
            print("Calculating detailed metrics...") # Debug print

            all_labels = []
            for images, labels in test_ds:
                all_labels.append(labels.numpy())
            true_labels_int = np.concatenate(all_labels, axis=0)

            if test_ds.element_spec[1].shape[-1] > 1 and true_labels_int.ndim > 1:
                 true_labels_int = np.argmax(true_labels_int, axis=1)


            print("Getting model predictions for detailed metrics...") # Debug print
            predictions_raw = model.predict(test_ds)
            print("Predictions obtained.")

            predicted_labels_int = np.argmax(predictions_raw, axis=1)

            sorted_class_labels = test_ds.class_names # Use inferred names

            # Calculate Classification Report
            report = classification_report(true_labels_int, predicted_labels_int, target_names=sorted_class_labels, output_dict=True, zero_division=0)
            evaluation_metrics['classification_report'] = report
            print("Classification report calculated.")

            # Calculate Confusion Matrix
            cm = confusion_matrix(true_labels_int, predicted_labels_int)
            evaluation_metrics['confusion_matrix'] = cm.tolist()
            evaluation_metrics['confusion_matrix_labels'] = sorted_class_labels
            print("Confusion matrix calculated.")

            evaluation_metrics['inferred_class_names'] = sorted_class_labels

            evaluation_metrics['status'] = 'success'

            # 6. Return results as JSON
            print("Returning evaluation metrics as JSON.")
            return jsonify(evaluation_metrics)

        except Exception as e:
            print(f"Error during model evaluation or metric calculation: {e}")
            return jsonify({"error": f"Error during model evaluation or metric calculation: {e}"}), 500


    except Exception as e:
        # Catches errors before specific blocks (e.g., tempfile.mkdtemp, finding root folder)
        print(f"An unexpected error occurred during initial processing: {e}")
        return jsonify({"error": f"An unexpected server error occurred during initial processing: {e}"}), 500


    finally:
        # 7. Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                print(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir)
                print("Cleanup complete.")
            except Exception as e:
                print(f"Error cleaning up temporary directory {temp_dir}: {e}")
                # Log this, but don't block the response.

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