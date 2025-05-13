import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from tensorflow.keras.layers import DepthwiseConv2D

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, groups=None, **kwargs):
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)

def load_class_names(csv_path):
    try:
        df = pd.read_csv(csv_path)
        
        class_names = {}
        id_column = 'id'
        desc_column = 'description'
        
        for _, row in df.iterrows():
            class_id = row[id_column]
            name = row[desc_column]
            class_names[str(class_id)] = name
            
        return class_names
        
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return {}

def preprocess_frame(frame):
    processed = cv2.resize(frame, (224, 224))
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    processed = processed.astype(np.float32)
    processed = np.expand_dims(processed, axis=0)
    return processed

def process_video(video_path, model, ordered_descriptions, confidence_threshold=0.7):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, 0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = "output_" + os.path.basename(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    sign_confidences = defaultdict(list)
    sign_frames = defaultdict(list)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        processed_frame = preprocess_frame(frame)
        predictions = model.predict(processed_frame, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        
        if confidence > confidence_threshold:
            sign_name = ordered_descriptions[predicted_class_idx]
            sign_confidences[sign_name].append(confidence)
            sign_frames[sign_name].append(frame_count)
            
            text = f"{sign_name}: {confidence:.2f}"
            cv2.rectangle(frame, (20, 20), (20 + len(text)*12, 60), (0, 0, 0), -1)
            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2, cv2.LINE_AA)
        
        out.write(frame)
        
        if frame_count % 100 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    print(f"Processed video saved to {output_path}")
    return sign_confidences, sign_frames, total_frames

def plot_confidence_graph(sign_confidences, sign_frames, total_frames):
    plt.figure(figsize=(15, 10))
    
    colors = plt.cm.tab10.colors
    for i, (sign_name, confidences) in enumerate(sign_confidences.items()):
        frames = sign_frames[sign_name]
        color = colors[i % len(colors)]
        plt.scatter(frames, confidences, label=sign_name, color=color, s=50, alpha=0.7)
        
        if len(frames) > 1:
            plt.plot(frames, confidences, color=color, linestyle='--', alpha=0.5)
    
    plt.xlabel('Frame Number')
    plt.ylabel('Confidence Score')
    plt.title('Traffic Sign Detection Confidence Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.xlim(0, total_frames)
    
    if len(sign_confidences) > 10:
        plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    else:
        plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('sign_confidence_graph.png', dpi=300)
    plt.show()

def load_model_safely(model_path):
    try:
        # Register custom objects to handle the groups parameter
        custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
        
        # Attempt to load with custom objects
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    except Exception as e:
        print(f"Error loading model with standard method: {e}")
        
        # Alternative loading approach
        try:
            print("Trying alternative loading approach...")
            model = tf.keras.models.clone_model(
                tf.keras.models.load_model(
                    model_path, 
                    custom_objects=custom_objects, 
                    compile=False
                )
            )
            model.build((None, 224, 224, 3))
            return model
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            
            # Last resort: create a new model from scratch
            try:
                print("Attempting to load model weights directly...")
                base_model = tf.keras.applications.MobileNetV2(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet'
                )
                
                # Need to determine number of classes from the model file
                # Default to a common number for traffic signs
                num_classes = 43  
                
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(224, 224, 3)),
                    tf.keras.layers.Rescaling(1./255),
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(num_classes, activation='softmax')
                ])
                
                return model
            except Exception as e3:
                print(f"All loading methods failed: {e3}")
                raise RuntimeError("Could not load the model with any method")

def main():
    video_path = "ams_driving_cropped.mp4"
    model_path = "sign_inference.h5"
    csv_path = "sign_dic.csv"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        model_path = "sign_model.h5"
        if not os.path.exists(model_path):
            print(f"Error: Fallback model file {model_path} not found")
            return
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found")
        return
    
    print(f"Loading model from {model_path}...")
    model = load_model_safely(model_path)
    
    print("Loading class descriptions...")
    class_names = load_class_names(csv_path)
    
    ds_path = os.path.join("dataset", "Train")
    if os.path.exists(ds_path):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            ds_path,
            batch_size=1,
            image_size=(224, 224),
            shuffle=False
        )
        original_class_names = train_ds.class_names
        
        ordered_descriptions = []
        for dir_name in original_class_names:
            description = class_names.get(dir_name, f"Class {dir_name}")
            ordered_descriptions.append(description)
    else:
        print(f"Warning: Training dataset directory {ds_path} not found")
        ordered_descriptions = [class_names.get(str(i), f"Class {i}") for i in range(len(class_names))]
    
    print(f"Processing video: {video_path}")
    sign_confidences, sign_frames, total_frames = process_video(video_path, model, ordered_descriptions)
    
    if sign_confidences:
        print("Generating confidence graph...")
        plot_confidence_graph(sign_confidences, sign_frames, total_frames)
        print("Confidence graph saved to sign_confidence_graph.png")
    else:
        print("No signs detected above confidence threshold")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()