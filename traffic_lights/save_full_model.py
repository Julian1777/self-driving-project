import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Concatenate, Lambda
from tensorflow.keras.applications import EfficientNetB0
import os

IMG_SIZE = (64,64)
INPUT_SHAPE = (64, 64, 3)
STATES = ["green", "red", "yellow", "off"]

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomRotation(0.25),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    #Color Jitter
    tf.keras.layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.2)),
    tf.keras.layers.Lambda(lambda x: tf.image.random_contrast(x, lower=0.8, upper=1.2)),
    tf.keras.layers.Lambda(lambda x: tf.image.random_saturation(x, lower=0.8, upper=1.2)),
    tf.keras.layers.Lambda(lambda x: tf.image.random_hue(x, max_delta=0.05))
])

def hsv_feature_extraction(image):
    # Convert to HSV colorspace
    hsv = tf.image.rgb_to_hsv(image)
    
    # Extract all channels
    h_channel = hsv[:, :, :, 0]  # Hue
    s_channel = hsv[:, :, :, 1]  # Saturation
    v_channel = hsv[:, :, :, 2]  # Value (brightness)
    
    # Get image dimensions
    batch_size = tf.shape(image)[0]
    height = tf.shape(image)[1]
    width = tf.shape(image)[2]
    
    # Divide image into three vertical regions (top, middle, bottom)
    h_third = height // 3
    
    # Extract regions for all channels
    # Brightness (Value)
    top_v = v_channel[:, :h_third, :] 
    middle_v = v_channel[:, h_third:2*h_third, :]
    bottom_v = v_channel[:, 2*h_third:, :]
    
    # Saturation
    top_s = s_channel[:, :h_third, :] 
    middle_s = s_channel[:, h_third:2*h_third, :]
    bottom_s = s_channel[:, 2*h_third:, :]
    
    # Hue
    top_h = h_channel[:, :h_third, :] 
    middle_h = h_channel[:, h_third:2*h_third, :]
    bottom_h = h_channel[:, 2*h_third:, :]
    
    # --- BRIGHTNESS FEATURES ---
    # Keep all existing brightness features
    top_brightness = tf.reduce_mean(top_v, axis=[1, 2])
    middle_brightness = tf.reduce_mean(middle_v, axis=[1, 2])
    bottom_brightness = tf.reduce_mean(bottom_v, axis=[1, 2])
    
    overall_brightness = tf.reduce_mean(v_channel, axis=[1, 2])
    
    epsilon = 1e-7
    rel_top_brightness = top_brightness / (overall_brightness + epsilon)
    rel_middle_brightness = middle_brightness / (overall_brightness + epsilon)
    rel_bottom_brightness = bottom_brightness / (overall_brightness + epsilon)
    
    max_top_brightness = tf.reduce_max(top_v, axis=[1, 2])
    max_middle_brightness = tf.reduce_max(middle_v, axis=[1, 2])
    max_bottom_brightness = tf.reduce_max(bottom_v, axis=[1, 2])
    
    var_top_brightness = tf.math.reduce_variance(tf.reshape(top_v, [batch_size, -1]), axis=1)
    var_middle_brightness = tf.math.reduce_variance(tf.reshape(middle_v, [batch_size, -1]), axis=1)
    var_bottom_brightness = tf.math.reduce_variance(tf.reshape(bottom_v, [batch_size, -1]), axis=1)
    
    # --- SATURATION FEATURES ---
    # Average saturation per region
    top_saturation = tf.reduce_mean(top_s, axis=[1, 2])
    middle_saturation = tf.reduce_mean(middle_s, axis=[1, 2])
    bottom_saturation = tf.reduce_mean(bottom_s, axis=[1, 2])
    
    # Maximum saturation (helps detect vivid colors)
    max_top_saturation = tf.reduce_max(top_s, axis=[1, 2])
    max_middle_saturation = tf.reduce_max(middle_s, axis=[1, 2])
    max_bottom_saturation = tf.reduce_max(bottom_s, axis=[1, 2])
    
    # --- HUE FEATURES ---
    # Calculate histograms of hue values in each region
    # Traffic light colors have specific hue ranges:
    # Red: ~0.0 or ~1.0
    # Yellow: ~0.1-0.2
    # Green: ~0.3-0.4
    
    red_mask1 = tf.logical_and(
        tf.greater_equal(h_channel, 0.0),
        tf.less_equal(h_channel, 10.0/180.0)
    )
    red_mask2 = tf.logical_and(
        tf.greater_equal(h_channel, 170.0/180.0),
        tf.less_equal(h_channel, 1.0)
    )
    
    # Also require minimum saturation and value
    red_mask1 = tf.logical_and(
        red_mask1, 
        tf.logical_and(
            tf.greater_equal(s_channel, 100.0/255.0),
            tf.greater_equal(v_channel, 100.0/255.0)
        )
    )
    red_mask2 = tf.logical_and(
        red_mask2, 
        tf.logical_and(
            tf.greater_equal(s_channel, 100.0/255.0),
            tf.greater_equal(v_channel, 100.0/255.0)
        )
    )
    
    red_mask = tf.logical_or(red_mask1, red_mask2)
    
    # Yellow mask (20-30 in OpenCV scale)
    yellow_mask = tf.logical_and(
        tf.logical_and(
            tf.greater_equal(h_channel, 20.0/180.0),
            tf.less_equal(h_channel, 30.0/180.0)
        ),
        tf.logical_and(
            tf.greater_equal(s_channel, 100.0/255.0),
            tf.greater_equal(v_channel, 100.0/255.0)
        )
    )
    
    # Green mask (40-80 in OpenCV scale)
    green_mask = tf.logical_and(
        tf.logical_and(
            tf.greater_equal(h_channel, 40.0/180.0),
            tf.less_equal(h_channel, 80.0/180.0)
        ),
        tf.logical_and(
            tf.greater_equal(s_channel, 100.0/255.0),
            tf.greater_equal(v_channel, 100.0/255.0)
        )
    )
    
    # Convert boolean masks to float
    red_mask = tf.cast(red_mask, tf.float32)
    yellow_mask = tf.cast(yellow_mask, tf.float32)
    green_mask = tf.cast(green_mask, tf.float32)
    
    # Count pixels with specific hues in each region
    top_red_ratio = tf.reduce_mean(red_mask[:, :h_third, :], axis=[1, 2])
    middle_red_ratio = tf.reduce_mean(red_mask[:, h_third:2*h_third, :], axis=[1, 2])
    bottom_red_ratio = tf.reduce_mean(red_mask[:, 2*h_third:, :], axis=[1, 2])
    
    top_yellow_ratio = tf.reduce_mean(yellow_mask[:, :h_third, :], axis=[1, 2])
    middle_yellow_ratio = tf.reduce_mean(yellow_mask[:, h_third:2*h_third, :], axis=[1, 2])
    bottom_yellow_ratio = tf.reduce_mean(yellow_mask[:, 2*h_third:, :], axis=[1, 2])
    
    top_green_ratio = tf.reduce_mean(green_mask[:, :h_third, :], axis=[1, 2])
    middle_green_ratio = tf.reduce_mean(green_mask[:, h_third:2*h_third, :], axis=[1, 2])
    bottom_green_ratio = tf.reduce_mean(green_mask[:, 2*h_third:, :], axis=[1, 2])
    
    # Stack all features
    hsv_features = tf.stack([
        # Original brightness features
        top_brightness, middle_brightness, bottom_brightness,
        rel_top_brightness, rel_middle_brightness, rel_bottom_brightness,
        max_top_brightness, max_middle_brightness, max_bottom_brightness,
        var_top_brightness, var_middle_brightness, var_bottom_brightness,
        overall_brightness,
        
        # Saturation features
        top_saturation, middle_saturation, bottom_saturation,
        max_top_saturation, max_middle_saturation, max_bottom_saturation,
        
        # Hue distribution features
        top_red_ratio, middle_red_ratio, bottom_red_ratio,
        top_yellow_ratio, middle_yellow_ratio, bottom_yellow_ratio,
        top_green_ratio, middle_green_ratio, bottom_green_ratio
    ], axis=1)

    hsv_features = tf.ensure_shape(hsv_features, (None, 28))
    
    return hsv_features

def build_traffic_light_model():
    inputs = Input(shape=INPUT_SHAPE)
    augmented = data_augmentation(inputs)

    base_model = EfficientNetB0(include_top=False, input_tensor=augmented, weights='imagenet')
    base_model.trainable = False  # freeze during initial training

    x = GlobalAveragePooling2D()(base_model.output)
    cnn_features = Dense(128, activation='relu')(x)
    cnn_features = Dropout(0.5)(cnn_features)

    hsv_features = Lambda(hsv_feature_extraction)(augmented)
    hsv_branch = Dense(32, activation='relu')(hsv_features)

    combined = Concatenate()([cnn_features, hsv_branch])
    outputs = Dense(len(STATES), activation='softmax', dtype='float32')(combined)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_inference_model():
    inputs = Input(shape=INPUT_SHAPE)
    
    base_model = EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')
    base_model.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    cnn_features = Dense(128, activation='relu')(x)
    cnn_features = Dropout(0.5)(cnn_features)
    
    hsv_features = Lambda(hsv_feature_extraction)(inputs)
    hsv_branch = Dense(32, activation='relu')(hsv_features)
    
    combined = Concatenate()([cnn_features, hsv_branch])
    outputs = Dense(len(STATES), activation='softmax', dtype='float32')(combined)
    
    return Model(inputs=inputs, outputs=outputs)


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
tf.saved_model.save(inference_model, saved_model_path)
print("Model saved successfully in SavedModel format")

try:
    print("Attempting to load the saved model...")
    loaded_model = tf.saved_model.load(saved_model_path)
    print("Model loaded successfully!")
    
    infer = loaded_model.signatures["serving_default"]
    print("Model signature verified. Saving was successful.")
except Exception as e:
    print(f"Error loading model: {e}")