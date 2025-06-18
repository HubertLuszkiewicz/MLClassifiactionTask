import tensorflow as tf
import numpy as np

# Import the ResNet50 model class
from tensorflow.keras.applications import ResNet50
# Import the utility functions specific to ResNet50 for preprocessing and decoding predictions
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

MODEL_SAVE_PATH = "../trained_models/resnet.h5"
def save_resnet50():
    model = ResNet50(weights='imagenet')
    model.save(MODEL_SAVE_PATH)


def load_resnet50():
    loaded_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    return loaded_model

# img_path = 'cat_0837.jpg'
# # This should match the input size your model expects (e.g., 224x224 for ResNet50)
# TARGET_SIZE = (224, 224)
# img = tf.keras.utils.load_img(img_path, target_size=TARGET_SIZE)
# img_array = tf.keras.utils.img_to_array(img)
# img_array_batched = np.expand_dims(img_array, axis=0)
# processed_image_array = preprocess_input(img_array_batched)
# predictions = model.predict(processed_image_array)

# decoded_predictions = decode_predictions(predictions, top=5)[0] # [0] because we had batch size 1

# print("\nPredictions for the image:")
# for imagenet_id, class_name, score in decoded_predictions:
#     # imagenet_id is like 'n02129165', class_name is 'lion', score is the probability
#     print(f"- {class_name} ({score:.2f})")