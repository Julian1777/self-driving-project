import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MaxPooling2D, Lambda, Conv2D, BatchNormalization, Flatten, Concatenate
import os

IMG_SIZE = (64,64)
INPUT_SHAPE = (64, 64, 3)
STATES = ["red", "yellow", "green"]
CROP_LEFT_RIGHT = 12
CROP_TOP_BOTTOM = 3

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomRotation(0.25),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.2)),
    tf.keras.layers.Lambda(lambda x: tf.image.random_contrast(x, lower=0.8, upper=1.2)),
    tf.keras.layers.Lambda(lambda x: tf.image.random_saturation(x, lower=0.8, upper=1.2)),
    tf.keras.layers.Lambda(lambda x: tf.image.random_hue(x, max_delta=0.05))
])

def extract_brightness_features(image):
    hsv = tf.image.rgb_to_hsv(image)
    v_channel = hsv[:, :, :, 2]
    cropped_v = v_channel[:, CROP_TOP_BOTTOM:tf.shape(image)[1]-CROP_TOP_BOTTOM, 
                         CROP_LEFT_RIGHT:tf.shape(image)[2]-CROP_LEFT_RIGHT]
    row_brightness = tf.reduce_mean(cropped_v, axis=2)
    return row_brightness

def build_traffic_light_model():
    inputs = Input(shape=INPUT_SHAPE, name="image_input")

    brightness = Lambda(extract_brightness_features, name="brightness_vec")(inputs)
    b = Flatten(name="flatten_brightness")(brightness)
    b = Dense(32, activation="relu", name="dense_brightness")(b)
    b = Dropout(0.3, name="drop_brightness")(b)

    x = Conv2D(32, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = BatchNormalization(name="bn1")(x)
    x = MaxPooling2D(2, name="pool1")(x)

    x = Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = BatchNormalization(name="bn2")(x)
    x = MaxPooling2D(2, name="pool2")(x)

    x = Conv2D(128, 3, padding="same", activation="relu", name="conv3")(x)
    x = BatchNormalization(name="bn3")(x)
    x = MaxPooling2D(2, name="pool3")(x)

    x = Flatten(name="flatten_cnn")(x)
    x = Dense(128, activation="relu", name="dense_cnn")(x)
    x = Dropout(0.5, name="drop_cnn")(x)

    combined = Concatenate(name="concat")([x, b])
    outputs = Dense(len(STATES), activation="softmax", name="class_output")(combined)

    model = Model(inputs, outputs, name="traffic_light_classifier")
    return model

def build_inference_model():
    inputs = Input(shape=INPUT_SHAPE, name="image_input")

    brightness = Lambda(extract_brightness_features, name="brightness_vec")(inputs)
    b = Flatten(name="flatten_brightness")(brightness)
    b = Dense(32, activation="relu", name="dense_brightness")(b)
    b = Dropout(0.3, name="drop_brightness")(b)

    x = Conv2D(32, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = BatchNormalization(name="bn1")(x)
    x = MaxPooling2D(2, name="pool1")(x)

    x = Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = BatchNormalization(name="bn2")(x)
    x = MaxPooling2D(2, name="pool2")(x)

    x = Conv2D(128, 3, padding="same", activation="relu", name="conv3")(x)
    x = BatchNormalization(name="bn3")(x)
    x = MaxPooling2D(2, name="pool3")(x)

    x = Flatten(name="flatten_cnn")(x)
    x = Dense(128, activation="relu", name="dense_cnn")(x)
    x = Dropout(0.5, name="drop_cnn")(x)

    combined = Concatenate(name="concat")([x, b])
    outputs = Dense(len(STATES), activation="softmax", name="class_output")(combined)

    return Model(inputs, outputs, name="traffic_light_classifier")

print("Building training model...")
train_model = build_traffic_light_model()

print("Building inference model...")
inference_model = build_inference_model()

checkpoint_path = "traffic_light_classification_checkpoint.h5"
if os.path.exists("traffic_light_classification_weights.h5"):
    checkpoint_path = "traffic_light_classification_weights.h5"

print(f"Loading weights from {checkpoint_path}...")
train_model.load_weights(checkpoint_path)

print("Transferring weights to inference model...")
train_layers = [layer.name for layer in train_model.layers]
inference_layers = [layer.name for layer in inference_model.layers]

for i, layer in enumerate(inference_model.layers):
    if layer.name in train_layers:
        for train_layer in train_model.layers:
            if train_layer.name == layer.name:
                inference_model.layers[i].set_weights(train_layer.get_weights())
                print(f"Copied weights for layer: {layer.name}")
                break

saved_model_path = "traffic_light_classification_savedmodel"
print(f"Saving model to {saved_model_path}...")

@tf.function(input_signature=[tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32)])
def serving_function(input_imgs):
    # Manually extract brightness features
    hsv = tf.image.rgb_to_hsv(input_imgs)
    v_channel = hsv[:, :, :, 2]
    cropped_v = v_channel[:, CROP_TOP_BOTTOM:tf.shape(input_imgs)[1]-CROP_TOP_BOTTOM, 
                         CROP_LEFT_RIGHT:tf.shape(input_imgs)[2]-CROP_LEFT_RIGHT]
    row_brightness = tf.reduce_mean(cropped_v, axis=2)
    
    # Print shape for debugging
    # tf.print("row_brightness shape:", tf.shape(row_brightness))
    
    # Fix the reshape - row_brightness has shape [batch_size, height]
    b = tf.reshape(row_brightness, [tf.shape(input_imgs)[0], -1])
    
    # Get brightness branch weights
    brightness_dense = inference_model.get_layer("dense_brightness")
    brightness_dropout = inference_model.get_layer("drop_brightness")
    
    # Process brightness manually
    b = brightness_dense(b)
    b = brightness_dropout(b, training=False)
    
    # Get CNN branch outputs by running partial model
    x = inference_model.get_layer("conv1")(input_imgs)
    x = inference_model.get_layer("bn1")(x)
    x = inference_model.get_layer("pool1")(x)
    x = inference_model.get_layer("conv2")(x)
    x = inference_model.get_layer("bn2")(x)
    x = inference_model.get_layer("pool2")(x)
    x = inference_model.get_layer("conv3")(x)
    x = inference_model.get_layer("bn3")(x)
    x = inference_model.get_layer("pool3")(x)
    x = inference_model.get_layer("flatten_cnn")(x)
    x = inference_model.get_layer("dense_cnn")(x)
    x = inference_model.get_layer("drop_cnn")(x, training=False)
    
    # Manual concatenation
    combined = tf.concat([x, b], axis=1)
    
    # Final classification
    outputs = inference_model.get_layer("class_output")(combined)
    
    return {"predictions": outputs}

tf.saved_model.save(
    inference_model, 
    "traffic_light_classification_savedmodel",
    signatures={"serving_default": serving_function}
)

print("Model saved successfully in SavedModel format")

try:
    print("Attempting to load the saved model...")
    loaded_model = tf.saved_model.load(saved_model_path)
    print("Model loaded successfully!")
    
    infer = loaded_model.signatures["serving_default"]
    print("Model signature verified. Saving was successful.")
except Exception as e:
    print(f"Error loading model: {e}")