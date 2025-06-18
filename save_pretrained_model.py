# save_pretrained_model.py
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import os

# Define where to save the model *inside the Docker container*
# This should match where your Flask app expects to find models
MODEL_SAVE_DIR_INSIDE_CONTAINER = '/app/trained_models'
MODEL_SAVE_PATH_INSIDE_CONTAINER = os.path.join(MODEL_SAVE_DIR_INSIDE_CONTAINER, 'resnet50_imagenet.keras')

# Create the directory if it doesn't exist
os.makedirs(MODEL_SAVE_DIR_INSIDE_CONTAINER, exist_ok=True)

print(f"Attempting to load and save pre-trained ResNet50 to {MODEL_SAVE_PATH_INSIDE_CONTAINER}")

try:
    # Load the pre-trained model (this downloads weights if not cached locally on the build machine)
    print("Loading ResNet50 with ImageNet weights...")
    model = ResNet50(weights='imagenet')
    print("Model loaded.")

    # Save the model in the recommended Keras format
    model.save(MODEL_SAVE_PATH_INSIDE_CONTAINER)
    print(f"Pre-trained ResNet50 model saved successfully to {MODEL_SAVE_PATH_INSIDE_CONTAINER}")

except Exception as e:
    print(f"Error saving model: {e}")
    # Exit with a non-zero status code to indicate failure during build
    exit(1)
