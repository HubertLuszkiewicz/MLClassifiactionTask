from flask import Flask, request, jsonify, render_template
import os
import shutil # For temporary directory cleanup
import tempfile # For creating temporary directories
import json # For saving class names
import numpy as np
import io # Make sure this is imported
# Assume necessary imports for TensorFlow and Keras are available
import tensorflow as tf
from tensorflow.keras.models import Model, load_model # Need Model for building, load_model if loading base from file
from tensorflow.keras.applications import ResNet50 # Or other base model
from tensorflow.keras.applications.resnet50 import preprocess_input # Preprocessing for ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # Layers for the new top
from tensorflow.keras.optimizers import Adam # Optimizer
from tensorflow.keras.losses import CategoricalCrossentropy # Loss function
from tensorflow.keras.preprocessing import image_dataset_from_directory # Data loading utility
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import load_img, img_to_array

def build_finetuned_model(num_classes, base_model_architecture='ResNet50', input_shape=(224, 224, 3)):
    print(f"Building fine-tuned model for {num_classes} classes based on {base_model_architecture}...")

    # --- 1. Load the pre-trained base model ---
    if base_model_architecture == 'ResNet50':
        # weights='imagenet': Use ImageNet weights
        # include_top=False: Crucially, remove the original 1000-class top layer
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        # You might load a pre-trained ResNet18 if you have its weights file
        # base_model = load_model('/path/to/your/resnet18_base.h5', compile=False) # Example if saving just the base

        # Freeze the base model layers initially for phase 1 training
        for layer in base_model.layers:
            layer.trainable = False
    # Add elif for other architectures if needed
    else:
        raise ValueError(f"Unsupported base model architecture: {base_model_architecture}")

    print("Base model loaded and frozen.")

    # --- 2. Add new classification layers on top ---
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Global pooling before dense layers
    # Add a dropout layer for regularization if needed
    # x = tf.keras.layers.Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x) # Final dense layer with NUM_CLASSES units

    # --- 3. Create the full model ---
    model = Model(inputs=base_model.input, outputs=predictions)

    print("Fine-tuned model architecture created.")
    return model

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
    # Initialize temporary directory variable
    temp_dir = None
    training_status = {"status": "processing"}

    try:
        print("--- Received POST request to /train ---") # Debug print

        # 1. Receive and Save Training Data (Folder)
        print("Attempting to receive and save training data folder...") # Debug print
        try:
            training_files = request.files.getlist('trainingData')
            # We also expect a selected model path (for the base model to fine-tune)
            # This might come from a separate dropdown for base models or be hardcoded
            # For simplicity, let's assume we are fine-tuning a standard ResNet50 ImageNet here.
            # If you need to select a base model file, you'd add an input for it.

            if not training_files:
                print("Error: No training data files received.") # Debug print
                return jsonify({"error": "No training data files received."}), 400

            temp_dir = tempfile.mkdtemp(prefix='train_data_')
            print(f"Saving {len(training_files)} training files to temporary directory: {temp_dir}") # Debug print

            for file_storage in training_files:
                # file_storage.filename contains the relative path (e.g., 'my_dataset/class_a/image.jpg')
                save_path = os.path.join(temp_dir, file_storage.filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                file_storage.save(save_path)

            print("Finished saving training data files.") # Debug print

        except Exception as e:
            print(f"Error during training data reception or saving: {e}")
            return jsonify({"error": f"Error receiving or saving training data files: {e}"}), 500

        # --- Find the root directory within the temporary directory ---
        # (Similar logic as in evaluate_model)
        dataset_root_in_temp = None
        try:
            items_in_temp_dir = os.listdir(temp_dir)
            if len(items_in_temp_dir) == 1 and os.path.isdir(os.path.join(temp_dir, items_in_temp_dir[0])):
                 dataset_root_name = items_in_temp_dir[0]
                 dataset_root_in_temp = os.path.join(temp_dir, dataset_root_name)
                 print(f"Identified dataset root folder inside temp dir: {dataset_root_in_temp}") # Debug print
            else:
                 print(f"Error: Unexpected structure inside temp directory. Expected a single root folder.") # Debug print
                 print(f"Contents: {items_in_temp_dir}") # Debug print
                 return jsonify({"error": "Uploaded training data has an unexpected structure. Please upload a single root folder containing class subfolders."}), 400

        except Exception as e:
            print(f"Error identifying dataset root folder in temp dir: {e}")
            return jsonify({"error": f"Server error processing uploaded folder structure: {e}"}), 500


        # 2. Create Training and Validation Datasets
        print("Attempting to create training and validation datasets...") # Debug print
        try:
            # image_dataset_from_directory for training data
            # Use a validation_split and subset for creating train/validation sets
            train_ds = image_dataset_from_directory(
                dataset_root_in_temp, # Point to the identified root folder
                labels='inferred',
                label_mode='categorical', # Or 'int', match your model's expected output
                image_size=(224, 224), # Match model input size
                batch_size=32,
                shuffle=True,
                seed=123, # Use a seed for reproducible split
                validation_split=0.2, # Use 20% for validation
                subset='training'
            )

            val_ds = image_dataset_from_directory(
                dataset_root_in_temp,
                labels='inferred',
                label_mode='categorical',
                image_size=(224, 224),
                batch_size=32,
                shuffle=False, # Don't shuffle validation
                seed=123,
                validation_split=0.2,
                subset='validation'
            )

            # Get the class names from the training dataset
            class_names = train_ds.class_names
            num_classes = len(class_names)
            print(f"Detected {num_classes} classes: {class_names}") # Debug print

            if num_classes == 0:
                 print("Error: No classes inferred from training data folders.") # Debug print
                 return jsonify({"error": "No valid image files found in class subfolders for training."}), 400


            # Apply preprocessing
            preprocess_func = preprocess_input # Use the appropriate preprocess function for your base model

            train_ds = train_ds.map(lambda x, y: (preprocess_func(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(lambda x, y: (preprocess_func(x), y), num_parallel_calls=tf.data.AUTOTUNE)

            # Optimize data loading
            train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

            print(f"Created training dataset ({len(train_ds)} batches) and validation dataset ({len(val_ds)} batches).") # Debug print

        except Exception as e:
            print(f"Error creating datasets from {dataset_root_in_temp}: {e}")
            return jsonify({"error": f"Error creating training/validation datasets. Ensure structure is class_name/image.jpg: {e}"}), 400


        # 3. Build the Fine-tuned Model
        print("Attempting to build the fine-tuned model...") # Debug print
        try:
            # Build the model using the number of classes detected
            model = build_finetuned_model(num_classes=num_classes, base_model_architecture='ResNet50') # Use your build function
            print("Fine-tuned model built.") # Debug print

        except Exception as e:
             print(f"Error building model: {e}")
             return jsonify({"error": f"Error building fine-tuned model: {e}"}), 500


        # 4. Compile the Model (Phase 1: Train top layers)
        print("Compiling model for Phase 1 training...") # Debug print
        try:
            model.compile(
                optimizer=Adam(learning_rate=1e-3), # Higher learning rate for new layers
                loss=CategoricalCrossentropy(),
                metrics=['accuracy']
            )
            print("Model compiled for Phase 1.") # Debug print
        except Exception as e:
             print(f"Error compiling model (Phase 1): {e}")
             return jsonify({"error": f"Error compiling model for training: {e}"}), 500


        # 5. Train the Model (Phase 1)
        print("Starting Phase 1 training (top layers)...") # Debug print
        try:
            history_phase1 = model.fit(
                train_ds,
                epochs=5, # Define number of epochs for phase 1
                validation_data=val_ds,
                verbose=1 # Set verbose level (1 for progress bar in logs)
            )
            print("Phase 1 training finished.") # Debug print

        except Exception as e:
             print(f"Error during Phase 1 training: {e}")
             return jsonify({"error": f"Error during model training (Phase 1): {e}"}), 500

        # --- Optional: Phase 2 Fine-tuning ---
        # This takes longer and requires more resources. You might skip it
        # or make it optional via the UI for simplicity initially.
        # If you include it, remember to unfreeze layers and re-compile with a lower LR.

        # print("Starting Phase 2 fine-tuning...")
        # try:
        #     # Unfreeze base model layers (or parts of it)
        #     for layer in model.layers[0].layers: # model.layers[0] is the base_model
        #         layer.trainable = True
        #     print("Base model layers unfrozen.")

        #     # Re-compile with lower learning rate
        #     model.compile(
        #         optimizer=Adam(learning_rate=1e-5), # Lower learning rate
        #         loss=CategoricalCrossentropy(),
        #         metrics=['accuracy']
        #     )
        #     print("Model compiled for Phase 2.")

        #     history_phase2 = model.fit(
        #         train_ds,
        #         epochs=5 + 5, # Total epochs (Phase 1 + Phase 2)
        #         initial_epoch=5, # Start from end of Phase 1
        #         validation_data=val_ds,
        #         verbose=1
        #     )
        #     print("Phase 2 fine-tuning finished.")

        # except Exception as e:
        #      print(f"Error during Phase 2 fine-tuning: {e}")
        #      return jsonify({"error": f"Error during model fine-tuning (Phase 2): {e}"}), 500
        # --- End Optional Phase 2 ---


        # 6. Save the Trained Model and Class Names
        print("Attempting to save trained model and class names...") # Debug print
        try:
            # Ensure the save directory exists
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

            # Define a name for the saved model (e.g., timestamp or user-provided name)
            # For now, let's use a simple name or include a timestamp
            import time
            timestamp = int(time.time())
            model_save_name = f'finetuned_model_{timestamp}.keras' # Or .h5
            model_save_path = os.path.join(MODEL_SAVE_DIR, model_save_name)

            # Save the model
            model.save(model_save_path)
            print(f"Trained model saved to {model_save_path}") # Debug print

            # Save the class names mapping (crucial for inference and evaluation)
            class_names_save_path = os.path.join(MODEL_SAVE_DIR, f'class_names_{timestamp}.json') # Save with corresponding name
            with open(class_names_save_path, 'w') as f:
                 json.dump(class_names, f)
            print(f"Class names mapping saved to {class_names_save_path}") # Debug print

            # You might want to store info about the saved model (path, name, class_names path)
            # in a database or a manifest file if you have many models.
            # For now, we just print paths.

            training_status['status'] = 'success'
            training_status['message'] = 'Model training completed successfully!'
            training_status['saved_model_path'] = model_save_path
            training_status['saved_class_names_path'] = class_names_save_path
            # Include final metrics if you want (e.g., from history_phase1.history)

            # 7. Return success response with paths to saved files
            print("Returning training success response.") # Debug print
            return jsonify(training_status)

        except Exception as e:
            print(f"Error during model or class names saving: {e}")
            # This is a critical error after training
            training_status['status'] = 'error'
            training_status['message'] = f'Model training completed, but failed to save model or class names: {e}'
            return jsonify(training_status), 500 # Return 500 as it's a server error

    except Exception as e:
        # This catches errors that happen very early (e.g., tempfile, initial file receiving)
        print(f"An unexpected error occurred during initial training processing: {e}") # Debug print
        training_status['status'] = 'error'
        training_status['message'] = f'An unexpected server error occurred during initial processing: {e}'
        return jsonify(training_status), 500

    finally:
        # 8. Clean up the temporary data directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                print(f"Cleaning up temporary data directory: {temp_dir}") # Debug print
                shutil.rmtree(temp_dir)
                print("Cleanup complete.") # Debug print
            except Exception as e:
                print(f"Error cleaning up temporary directory {temp_dir}: {e}")
                # Log this error, but don't block the response.

@app.route('/evaluate', methods=["GET"])
def show_evaluation_page():
    return render_template('evaluate_model.html')

@app.route('/evaluate', methods=["POST"])
def evaluate_model():
    # Initialize temporary directory variable outside the try block
    temp_dir = None
    # Initialize dictionary to hold results
    evaluation_metrics = {"status": "processing"}
    # Initialize variable to hold the dataset's inferred class names
    test_dataset_class_names = None

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
            # The 'dir' parameter could restrict temp directories to a specific parent
            temp_dir = tempfile.mkdtemp(prefix='test_eval_')
            print(f"Saving {len(test_files)} test files to temporary directory: {temp_dir}") # Debug print

            for file_storage in test_files:
                # Use file_storage.filename directly as it contains the relative path from the upload
                save_path = os.path.join(temp_dir, file_storage.filename)
                # Ensure the subdirectory for this file exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Save the file
                file_storage.save(save_path)

            print("Finished saving test files.") # Debug print

        except Exception as e:
            print(f"Error during file reception or saving: {e}")
            return jsonify({"error": f"Error receiving or saving test files: {e}"}), 500 # Use 500 for server-side issue


        # 2. Find the root directory within the temporary directory
        print("Attempting to find dataset root folder within temp dir...") # Debug print
        dataset_root_in_temp = None
        try:
            items_in_temp_dir = os.listdir(temp_dir)
            # We expect exactly one item, which is the root folder the user uploaded
            if len(items_in_temp_dir) == 1 and os.path.isdir(os.path.join(temp_dir, items_in_temp_dir[0])):
                 dataset_root_name = items_in_temp_dir[0]
                 dataset_root_in_temp = os.path.join(temp_dir, dataset_root_name)
                 print(f"Identified dataset root folder inside temp dir: {dataset_root_in_temp}") # Debug print
            else:
                 # This happens if the user uploaded individual files or multiple items at the top level
                 print(f"Error: Unexpected structure inside temp directory. Expected a single root folder.") # Debug print
                 print(f"Contents: {items_in_temp_dir}") # Debug print
                 return jsonify({"error": "Uploaded test data has an unexpected structure. Please upload a single root folder containing class subfolders (e.g., upload the 'mini_dataset' folder itself, not its contents)."}), 400 # Use 400 for bad client data

        except Exception as e:
            print(f"Error identifying dataset root folder in temp dir: {e}")
            return jsonify({"error": f"Server error processing uploaded folder structure: {e}"}), 500


        # 3. Load Model
        print("Attempting to load model...") # Debug print
        model = None # Initialize model variable
        try:
            selected_model_path = request.form.get('modelPath')
            if not selected_model_path:
                print("Error: No model path selected.") # Debug print
                return jsonify({"error": "No model path selected."}), 400

            # IMPORTANT SECURITY/VALIDATION: Check if the model path is safe!
            # Ensure path is within MODEL_SAVE_DIR.
            abs_model_save_dir = os.path.abspath(MODEL_SAVE_DIR)
            abs_selected_model_path = os.path.abspath(selected_model_path)

            # Check if the selected path is actually *inside* the designated model save directory
            if not abs_selected_model_path.startswith(abs_model_save_dir):
                 print(f"SECURITY ALERT: Attempted to load model outside MODEL_SAVE_DIR: {selected_model_path}")
                 return jsonify({"error": "Invalid model path specified."}), 400 # Use 400 for bad client input

            if not os.path.exists(abs_selected_model_path):
                print(f"Model file not found: {selected_model_path}") # Debug print
                return jsonify({"error": f"Model file not found at server path: {selected_model_path}"}), 404 # Use 404 if resource not found

            # Use your load_model_from_file function - It should load with compile=False
            model = load_model_from_file(abs_selected_model_path) # Assuming this function handles its own basic printing
            if model is None:
                 # load_model_from_file should print details about the load failure
                 # Its own error message is often sufficient, but adding a generic one here
                 print("load_model_from_file returned None.") # Debug print
                 return jsonify({"error": f"Failed to load model from {selected_model_path}. Check server logs for details."}), 500 # Use 500 for server error

            print("Model loaded successfully.") # Debug print

        except Exception as e: # Catch any unexpected errors *during* model loading (e.g., if load_model_from_file raises)
             print(f"Unexpected error during model loading process: {e}")
             return jsonify({"error": f"Unexpected error during model loading process: {e}"}), 500


        # 4. Create Test Dataset from the identified dataset root
        print("Attempting to create test dataset from the identified root...") # Debug print
        raw_test_ds = None # Initialize raw dataset variable
        test_ds = None # Initialize processed dataset variable
        try:
            # 1. Create the initial dataset object from the *correct* path
            raw_test_ds = image_dataset_from_directory(
                dataset_root_in_temp, # Point to the identified root folder (e.g., /tmp/.../mini_dataset)
                labels='inferred',
                label_mode='categorical', # Or 'int', MUST MATCH how the model was trained!
                image_size=(224, 224), # MUST match model's expected input size
                batch_size=32, # Batch size for evaluation
                shuffle=False # Don't shuffle evaluation data for consistent results
            )

            # 2. Check if any classes were inferred. This can fail if the dataset_root_in_temp
            # doesn't contain valid class subdirectories with images.
            if not raw_test_ds.class_names:
                 print("Error: No classes inferred from test data folders.") # Debug print
                 return jsonify({"error": "No valid image files found in class subfolders within the uploaded test data. Ensure structure is class_name/image.jpg"}), 400 # Use 400 for bad client data

            # 3. Get the class names *immediately* from the raw dataset object
            test_dataset_class_names = raw_test_ds.class_names # <-- GET CLASS NAMES HERE
            print(f"Test dataset inferred classes: {test_dataset_class_names}") # Debug print
            print(f"Inferred {len(test_dataset_class_names)} classes.") # Debug print

            # --- Validate Class Count vs. Model Output Shape ---
            # Check if the number of inferred classes matches the model's expected output shape.
            # This requires knowing the model's output shape. Assuming model output layer is Dense.
            # model.output_shape will be (None, num_classes)
            if model is not None and model.output_shape[-1] != len(test_dataset_class_names):
                 print(f"Error: Model expects {model.output_shape[-1]} classes, but inferred {len(test_dataset_class_names)} classes from test data.")
                 return jsonify({"error": f"Model expects {model.output_shape[-1]} classes, but uploaded test data contains {len(test_dataset_class_names)} classes based on folder names. Class count mismatch."}), 400 # Use 400

            # 4. Apply the necessary transformations and optimizations
            # Use raw_test_ds.map, raw_test_ds.cache, raw_test_ds.prefetch
            test_ds = raw_test_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE) # This is the _PrefetchDataset

            print(f"Created test dataset with {test_ds.cardinality().numpy()} batches.") # Debug print (cardinality requires iterating a bit)
            # Note: Getting total image count from PrefetchDataset can be tricky, batch count is more reliable here.

        except Exception as e:
            print(f"Error creating test dataset from {dataset_root_in_temp}: {e}")
            return jsonify({"error": f"Error creating test dataset from uploaded files: {e}. Ensure structure is class_name/image.jpg and images are valid."}), 400 # Use 400 for client data issue


        # 5. Compile the Model for Evaluation
        print("Attempting to compile model for evaluation...") # Debug print
        try:
            # Re-compile the model. Need a loss and metrics.
            # Match the loss type (CategoricalCrossentropy for categorical labels).
            model.compile(
                optimizer=Adam(), # Optimizer instance is required but its state isn't used by evaluate
                loss=CategoricalCrossentropy(), # Use the class instance if imported, or string 'categorical_crossentropy'
                metrics=['accuracy'] # Include 'accuracy' or other metrics you want evaluated
            )
            print("Model compiled for evaluation.") # Debug print
        except Exception as e:
             print(f"Error compiling model for evaluation: {e}")
             return jsonify({"error": f"Error compiling model for evaluation: {e}"}), 500 # Use 500 for server error


        # 6. Evaluate the Model and Calculate Detailed Metrics
        print("Attempting to evaluate model and calculate metrics...") # Debug print
        try:
            print("Starting core model evaluation (loss, accuracy)...") # Debug print
            # Use verbose=0 to prevent a progress bar in the console logs
            results_from_evaluate = model.evaluate(test_ds, verbose=0)
            print("Model evaluation finished.") # Debug print

            # Ensure results_from_evaluate is a list (model.evaluate returns a list if multiple metrics)
            if not isinstance(results_from_evaluate, list):
                 results_from_evaluate = [results_from_evaluate]

            # Combine metric names and results into the dictionary
            evaluation_metrics['core_metrics'] = dict(zip(model.metrics_names, results_from_evaluate))
            print("Core evaluation metrics:", evaluation_metrics['core_metrics']) # Debug print


            # --- Calculate detailed metrics (Precision, Recall, F1, Confusion Matrix) ---
            # This part doesn't strictly *require* compile(), but often done together
            print("Calculating detailed metrics...") # Debug print

            # Get true labels (integer indices). Iterate over the *final* optimized test_ds.
            all_labels = []
            for images, labels in test_ds:
                all_labels.append(labels.numpy())
            true_labels_int = np.concatenate(all_labels, axis=0)

            # Convert one-hot true labels to integer indices if label_mode='categorical' was used for the dataset
            if test_ds.element_spec[1].shape[-1] > 1 and true_labels_int.ndim > 1:
                 true_labels_int = np.argmax(true_labels_int, axis=1)


            # Get predictions (raw output, e.g., probabilities from softmax)
            # Iterate over the *final* optimized test_ds to get predictions
            print("Getting model predictions for detailed metrics...") # Debug print
            # Use model.predict on the dataset - Keras handles batching
            predictions_raw = model.predict(test_ds)
            print("Predictions obtained.") # Debug print


            # Convert predictions to integer indices (e.g., using argmax for softmax output)
            predicted_labels_int = np.argmax(predictions_raw, axis=1)

            # Use the class names inferred from the test dataset for metrics reporting
            sorted_class_labels = test_dataset_class_names # Use the variable obtained earlier

            # Calculate Classification Report
            # zero_division=0 handles potential cases with no true/predicted samples for a class gracefully
            report = classification_report(true_labels_int, predicted_labels_int, target_names=sorted_class_labels, output_dict=True, zero_division=0)
            evaluation_metrics['classification_report'] = report
            print("Classification report calculated.") # Debug print

            # Calculate Confusion Matrix
            cm = confusion_matrix(true_labels_int, predicted_labels_int)
            evaluation_metrics['confusion_matrix'] = cm.tolist() # Convert numpy array to list for JSON
            evaluation_metrics['confusion_matrix_labels'] = sorted_class_labels # Add class order
            print("Confusion matrix calculated.") # Debug print

            # Add the class names inferred from the test data folders to the final result
            evaluation_metrics['inferred_class_names'] = sorted_class_labels

            evaluation_metrics['status'] = 'success' # Update overall status


            # 7. Return results as JSON
            print("Returning evaluation metrics as JSON.") # Debug print
            return jsonify(evaluation_metrics)

        except Exception as e:
            print(f"Error during model evaluation or metric calculation: {e}")
            return jsonify({"error": f"Error during model evaluation or metric calculation: {e}"}), 500 # Use 500 for server error


    except Exception as e:
        # This catches errors that might happen *very* early, before specific blocks
        # (e.g., issues with tempfile.mkdtemp itself, initial file reading before saving loop starts)
        print(f"An unexpected error occurred during initial processing: {e}") # Debug print
        return jsonify({"error": f"An unexpected server error occurred during initial processing: {e}"}), 500 # Use 500


    finally:
        # 8. Clean up the temporary directory
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
    print("--- Received POST request to /classify ---") # Debug print

    # --- No longer checking global default model/class names here ---

    # 1. Get the selected model path from the form data
    print("Attempting to get selected model path from form...") # Debug print
    selected_model_path = request.form.get('modelPath')
    if not selected_model_path:
        print("Error: No model path selected in form data.") # Debug print
        # This should also be caught by client-side validation, but good to have
        return jsonify({"error": "No model selected for classification."}), 400 # 400 for bad client request


    # 2. Load the selected model from the file path
    model = None # Initialize model variable
    print(f"Attempting to load model from: {selected_model_path}") # Debug print
    try:
        # IMPORTANT SECURITY/VALIDATION: Check if the model path is safe!
        # Ensure path is within MODEL_SAVE_DIR to prevent directory traversal attacks.
        abs_model_save_dir = os.path.abspath(MODEL_SAVE_DIR)
        abs_selected_model_path = os.path.abspath(selected_model_path)

        # Check if the absolute path of the selected model starts with the absolute path of the models directory
        if not abs_selected_model_path.startswith(abs_model_save_dir):
             print(f"SECURITY ALERT: Attempted to load model outside MODEL_SAVE_DIR: {selected_model_path}")
             return jsonify({"error": "Invalid model path specified."}), 400 # Use 400 for bad client input

        if not os.path.exists(abs_selected_model_path):
            print(f"Model file not found at expected server path: {selected_model_path}") # Debug print
            return jsonify({"error": f"Model file not found at server path: {selected_model_path}. Check model selection."}), 404 # Use 404 if resource not found

        # Use your load_model_from_file function - It should handle basic loading errors
        # Note: load_model_from_file itself should ideally use compile=False
        model = load_model_from_file(abs_selected_model_path)
        if model is None:
             # load_model_from_file should print details about the load failure to server logs
             print("load_model_from_file returned None, indicating load failure.") # Debug print
             return jsonify({"error": f"Failed to load model from {selected_model_path}. Check server logs for detailed load errors."}), 500 # 500 for server error

        print("Model loaded successfully.") # Debug print

    except Exception as e: # Catch any unexpected errors *during* the model loading *process* itself
         print(f"Unexpected error during model loading process: {e}")
         return jsonify({"error": f"Unexpected error during model loading process: {e}"}), 500


    # 3. Load the corresponding Class Names Mapping file
    print("Attempting to load class names mapping...") # Debug print
    class_names = None # Initialize class names variable
    try:
        # Assume class_names.json is saved in the same directory as the model file
        model_dir = os.path.dirname(abs_selected_model_path)
        # --- ADJUST THIS LOGIC TO FIND THE CORRECT class_names.json FILE ---
        # This is a common point of error if the naming convention is complex or inconsistent.
        # A more robust approach: include class_names_path in /list_models JSON response.

        # Example logic assuming pattern like model_XXX.keras and class_names_XXX.json
        model_filename = os.path.basename(abs_selected_model_path) # e.g., finetuned_model_1750259108.keras
        class_names_filename = None

        if model_filename.endswith('.keras'):
             base_name = model_filename[:-len('.keras')] # e.g., finetuned_model_1750259108
             # Assuming class names file matches base name structure (replace start part)
             class_names_filename = base_name.replace('finetuned_model', 'class_names') + '.json'
        elif model_filename.endswith('.h5'):
             base_name = model_filename[:-len('.h5')] # e.g., my_model.h5
             # Assuming class names file matches base name structure
             class_names_filename = base_name.replace('finetuned_model', 'class_names') + '.json' # Adjust 'finetuned_model' part if your h5 names differ
        # Add more elif for other formats if needed
        else:
             print(f"Warning: Model filename '{model_filename}' doesn't end with .keras or .h5. Cannot infer class names filename based on pattern.") # Debug print
             # Handle models saved in SavedModel format (directories) - class_names.json might be at the root of the directory
             if os.path.isdir(abs_selected_model_path):
                  class_names_filename = 'class_names.json' # Common name within SavedModel dir


        class_names_path = None
        if class_names_filename:
            class_names_path = os.path.join(model_dir, class_names_filename) # Construct potential full path

        # Check if the constructed path exists and is a file
        if class_names_path and os.path.exists(class_names_path) and os.path.isfile(class_names_path):
            with open(class_names_path, 'r') as f:
                class_names = json.load(f) # Load the list from JSON
            print(f"Class names loaded from {class_names_path} ({len(class_names)} classes).") # Debug print
        else:
            print(f"Error: Class names file not found or path is invalid at {class_names_path}") # Debug print
            # Class names are essential for meaningful output
            return jsonify({"error": f"Class names mapping file not found for the selected model. Expected at {class_names_path}. Cannot decode prediction result."}), 404 # 404 for missing resource


    except Exception as e:
        print(f"Error loading class names file: {e}") # Debug print
        return jsonify({"error": f"Error loading class names file: {e}"}), 500 # 500 for server error

    # --- Crucial Check: Ensure class_names was successfully loaded and is a list ---
    if not isinstance(class_names, list) or not class_names:
         print(f"Error: Loaded class_names is not a valid non-empty list: {class_names}") # Debug print
         return jsonify({"error": "Invalid or empty class names mapping loaded."}), 500 # 500 for server error


    # 4. Check for and get the uploaded image file
    print("Attempting to get uploaded image file from request.files...") # Debug print
    # This check is redundant if the blocks above return errors, but good practice
    if 'image' not in request.files or not request.files['image']:
         print("Error: No 'image' file part in request.") # Debug print
         return jsonify({"error": "You must upload an image file with the name 'image'."}), 400 # 400 for bad client request

    file = request.files['image'] # Get the FileStorage object


    # Define the target image size (must match what your model expects)
    # *** ADJUST THIS IF YOUR MODEL EXPECTS A DIFFERENT SIZE ***
    TARGET_IMAGE_SIZE = (224, 224)


    # --- 5. Process the image and get prediction ---
    print(f"Processing image: {file.filename}") # Debug print
    try:
        # Load the image file from the FileStorage object's stream into bytes
        image_bytes = file.stream.read()
        # Wrap the bytes in an io.BytesIO object, which load_img understands
        image_stream = io.BytesIO(image_bytes)

        # Load the image from the io.BytesIO stream and resize it
        # Make sure Pillow is installed: pip install Pillow
        img = load_img(image_stream, target_size=TARGET_IMAGE_SIZE)
        print("Image loaded and resized.") # Debug print

        # Convert the PIL image object to a NumPy array
        img_array = img_to_array(img)
        print(f"Image converted to array with shape: {img_array.shape}") # Debug print

        # Add a batch dimension at the beginning. Keras models expect input as a batch.
        img_array_batched = np.expand_dims(img_array, axis=0)
        print(f"Added batch dimension, shape: {img_array_batched.shape}") # Debug print

        # Apply the model's preprocessing (scaling, centering, etc.)
        # *** MAKE SURE preprocess_input MATCHES YOUR MODEL ARCHITECTURE (ResNet50) ***
        processed_image_array = preprocess_input(img_array_batched)
        print("Image preprocessing complete.") # Debug print

        # Make a prediction
        # model.predict returns a numpy array of shape (batch_size, num_classes)
        print("Getting model prediction...") # Debug print
        predictions_raw = model.predict(processed_image_array) # Use the loaded 'model'
        print("Prediction obtained.") # Debug print
        # For a single image input, predictions_raw will have shape (1, num_classes)


        # Get the predicted class index (index with the highest probability)
        predicted_index = np.argmax(predictions_raw[0]) # [0] because batch size is 1
        print(f"Predicted index: {predicted_index}") # Debug print

        # Get the predicted class name using the loaded mapping (class_names)
        # Ensure predicted_index is within the bounds of the class_names list
        if 0 <= predicted_index < len(class_names): # Use the local 'class_names' variable
             predicted_label = class_names[predicted_index]
             print(f"Predicted label: {predicted_label}") # Debug print
        else:
             # This indicates a mismatch in the number of classes between the model output and the class_names file
             predicted_label = "Unknown (Index out of bounds)"
             print(f"Error: Predicted index {predicted_index} is out of bounds for loaded class names list with {len(class_names)} classes.") # Debug print
             # Consider returning an error here instead of a successful response with "Unknown" if this is critical
             # return jsonify({"error": "Class index mismatch between model output and class names mapping."}), 500


        # Get the confidence score for the top prediction
        # np.max finds the maximum value (the highest probability)
        confidence = float(np.max(predictions_raw[0])) # Use float() for JSON serialization
        print(f"Confidence: {confidence:.4f}") # Debug print with formatting


        # --- 6. Return success response ---
        success_response = {
            "message": "Prediction successful!",
            "filename": file.filename,
            "predicted_label": predicted_label,
            "confidence": confidence,
            # Optional: Include the list of all class names for reference in the UI
            "class_names": class_names # Use the local 'class_names' variable
        }
        print("Returning success response for classify.") # Debug print
        return jsonify(success_response), 200 # Return 200 OK

    except Exception as e:
        # Catch any error during image processing, prediction, or decoding
        print(f"Error during image processing or prediction: {e}") # Debug print the specific error
        return jsonify({"error": f"Error processing image or getting prediction: {e}"}), 500 # Return 500 for server error