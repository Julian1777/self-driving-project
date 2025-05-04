import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random

# Constants
PRIMARY_MODEL_PATH = "traffic_light_detection.h5"
CHECKPOINT_MODEL_PATH = "traffic_light_detection_checkpoint.h5"
IMG_SIZE = (224, 224)  # Standard YOLO input size
CONF_THRESHOLD = 0.25  # Lower threshold to see more detections
NUM_TEST_IMAGES = 6    # Number of test images to process

def yolo_loss(y_true, y_pred):
    coord_true = y_true[:, :2]  # x, y center
    size_true = y_true[:, 2:4]  # width, height
    conf_true = y_true[:, 4:5]  # confidence
    
    coord_pred = y_pred[:, :2]
    size_pred = y_pred[:, 2:4]
    conf_pred = y_pred[:, 4:5]
    
    object_mask = conf_true > 0.5
    object_mask = tf.cast(object_mask, dtype=tf.float32)
    
    lambda_coord = 5.0  # Higher weight for coordinates (position is important)
    lambda_size = 5.0   # Higher weight for size
    lambda_conf = 1.0   # Standard weight for confidence
    lambda_noobj = 0.5  # Lower weight for background (no object)
    
    coord_loss = lambda_coord * tf.reduce_sum(
        object_mask * tf.reduce_sum(tf.square(coord_true - coord_pred), axis=1, keepdims=True)
    ) / tf.maximum(tf.reduce_sum(object_mask), 1.0)
    
    size_true_sqrt = tf.sqrt(tf.maximum(size_true, 1e-10))
    size_pred_sqrt = tf.sqrt(tf.maximum(size_pred, 1e-10))
    size_loss = lambda_size * tf.reduce_sum(
        object_mask * tf.reduce_sum(tf.square(size_true_sqrt - size_pred_sqrt), axis=1, keepdims=True)
    ) / tf.maximum(tf.reduce_sum(object_mask), 1.0)
    
    conf_obj_loss = lambda_conf * tf.reduce_sum(
        object_mask * tf.square(conf_true - conf_pred)
    ) / tf.maximum(tf.reduce_sum(object_mask), 1.0)
    
    conf_noobj_loss = lambda_noobj * tf.reduce_sum(
        (1 - object_mask) * tf.square(conf_true - conf_pred)
    ) / tf.maximum(tf.reduce_sum(1 - object_mask), 1.0)
    
    total_loss = coord_loss + size_loss + conf_obj_loss + conf_noobj_loss
    
    return total_loss

def load_model():
    """Load the YOLO detection model"""
    # Check for both model files and use whichever is available
    if os.path.exists(PRIMARY_MODEL_PATH):
        model_path = PRIMARY_MODEL_PATH
        print(f"Loading primary model from {model_path}")
    elif os.path.exists(CHECKPOINT_MODEL_PATH):
        model_path = CHECKPOINT_MODEL_PATH
        print(f"Loading checkpoint model from {model_path}")
    else:
        raise FileNotFoundError("No model file found. Please check file paths.")
    
    custom_objects = {"yolo_loss": yolo_loss}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"Model loaded successfully!")
    return model

def process_image(image_path):
    """Process an image for YOLO detection"""
    # Read image
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Store original dimensions for later scaling
    original_h, original_w = img.shape[:2]
    
    # Convert to RGB (YOLO models typically trained on RGB)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Resize to model input size
    img_resized = cv.resize(img_rgb, IMG_SIZE)
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, 0)
    
    return img, img_rgb, img_batch, (original_w, original_h)

def get_ground_truth(image_path):
    """Get ground truth bounding box from YOLO label file"""
    # Only support val images
    val_dir = "./yolo_dataset/images/val"
    if val_dir not in image_path:
        print(f"Not a validation image: {image_path}")
        return None
    
    # Get label file path
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join("./yolo_dataset/labels/val", f"{base_name}.txt")
    
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return None
    
    if os.path.getsize(label_path) == 0:
        print(f"Label file is empty: {label_path}")
        return None
    
    # Read YOLO format labels (class x_center y_center width height)
    ground_truth_boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, x_center, y_center, width, height = map(float, parts)
                ground_truth_boxes.append([x_center, y_center, width, height])
    
    print(f"Found {len(ground_truth_boxes)} ground truth boxes in {label_path}")
    return ground_truth_boxes

def detect_traffic_lights(model, image_path):
    """Detect traffic lights in an image using simplified YOLO model"""
    # Process image
    original_img, rgb_img, img_batch, (original_w, original_h) = process_image(image_path)
    
    # Get ground truth if available
    ground_truth = get_ground_truth(image_path)
    
    # Run inference
    prediction = model.predict(img_batch, verbose=0)
    print(f"Prediction shape: {prediction.shape}")
    
    # Simple model outputs directly [x, y, w, h, confidence]
    # Extract the prediction
    x, y, w, h, confidence = prediction[0]
    
    print(f"\nPREDICTION:")
    print(f"  Box (normalized): x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f}, confidence={confidence:.4f}")
    
    # Only create a detection if confidence exceeds threshold
    filtered_boxes = []
    filtered_scores = []
    
    if confidence >= CONF_THRESHOLD:
        # Scale to original image dimensions
        x1 = int((x - w/2) * original_w)
        y1 = int((y - h/2) * original_h)
        x2 = int((x + w/2) * original_w)
        y2 = int((y + h/2) * original_h)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(original_w, x2)
        y2 = min(original_h, y2)
        
        filtered_boxes.append([x1, y1, x2, y2])
        filtered_scores.append(float(confidence))
        
        # Print pixel coordinates
        print(f"  Box (pixels): [x1={x1}, y1={y1}, x2={x2}, y2={y2}]")
    else:
        print(f"  No detection (confidence {confidence:.4f} below threshold {CONF_THRESHOLD})")
    
    # Visualize results
    plt.figure(figsize=(12, 10))
    plt.imshow(rgb_img)
    plt.axis('off')
    
    # Create a more informative title
    title = f"Traffic Light Detection - {os.path.basename(image_path)}"
    if ground_truth:
        title += f"\nGround Truth: {len(ground_truth)} boxes"
    if len(filtered_boxes) > 0:
        title += f" | Prediction: conf={confidence:.2f}"
    plt.title(title)
    
    # Draw predicted bounding boxes
    for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
        x1, y1, x2, y2 = box
        
        # Draw rectangle
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor='lime', linewidth=3)
        plt.gca().add_patch(rect)
        
        # Add label with confidence score
        plt.text(x1, y1-10, f"PREDICTION: {score:.2f}", 
                color='white', fontsize=12, 
                bbox=dict(facecolor='lime', alpha=0.8))
    
    # Draw ground truth boxes if available
    if ground_truth:
        print("\nGROUND TRUTH:")
        for i, box in enumerate(ground_truth):
            x_center, y_center, width, height = box
            
            # Print ground truth in normalized coordinates
            print(f"  Box {i+1}: x={x_center:.4f}, y={y_center:.4f}, w={width:.4f}, h={height:.4f}")
            
            # Convert to pixel coordinates for visualization
            x1 = int((x_center - width/2) * original_w)
            y1 = int((y_center - height/2) * original_h)
            x2 = int((x_center + width/2) * original_w)
            y2 = int((y_center + height/2) * original_h)
            
            # Print pixel coordinates
            print(f"  Box {i+1} (pixels): [x1={x1}, y1={y1}, x2={x2}, y2={y2}]")
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_w, x2)
            y2 = min(original_h, y2)
            
            # Draw rectangle with different color
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor='red', linewidth=3, linestyle='--')
            plt.gca().add_patch(rect)
            
            # Add label
            plt.text(x1, y2+15, f"GROUND TRUTH #{i+1}", 
                    color='white', fontsize=12, 
                    bbox=dict(facecolor='red', alpha=0.8))
    else:
        print("No ground truth bounding boxes available for this image")
    
    # Compare prediction with ground truth if available
    if ground_truth and len(filtered_boxes) > 0:
        print("\nCOMPARISON:")
        # For simplicity, compare with the first ground truth box if multiple exist
        gt_box = ground_truth[0]
        gt_x_center, gt_y_center, gt_width, gt_height = gt_box
        
        # Calculate IoU
        def calculate_iou(box1, box2):
            # Convert to [x1, y1, x2, y2] format
            b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
            b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
            b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
            b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
            
            # Calculate area of intersection
            inter_x1 = max(b1_x1, b2_x1)
            inter_y1 = max(b1_y1, b2_y1)
            inter_x2 = min(b1_x2, b2_x2)
            inter_y2 = min(b1_y2, b2_y2)
            
            if inter_x2 < inter_x1 or inter_y2 < inter_y1:
                return 0.0
            
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
            b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
            
            return inter_area / (b1_area + b2_area - inter_area)
        
        pred_box = [x, y, w, h]
        iou = calculate_iou(gt_box, pred_box)
        
        print(f"  IoU score: {iou:.4f}")
        print(f"  Center error (x, y): ({abs(gt_x_center - x):.4f}, {abs(gt_y_center - y):.4f})")
        print(f"  Size error (w, h): ({abs(gt_width - w):.4f}, {abs(gt_height - h):.4f})")
        
        # Add IoU information to plot
        plt.figtext(0.5, 0.01, f"IoU: {iou:.4f} | Center error: ({abs(gt_x_center - x):.4f}, {abs(gt_y_center - y):.4f}) | Size error: ({abs(gt_width - w):.4f}, {abs(gt_height - h):.4f})", 
                   ha="center", fontsize=12, 
                   bbox={"facecolor":"orange", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for IoU text at bottom
    plt.show()
    
    return {
        "num_detections": len(filtered_boxes),
        "boxes": filtered_boxes,
        "scores": filtered_scores,
        "raw_prediction": prediction[0].tolist(),
        "ground_truth": ground_truth
    }

# Main execution
if __name__ == "__main__":
    # Load model
    try:
        model = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    # Only use validation directory
    val_dir = "./yolo_dataset/images/val"
    
    if not os.path.isdir(val_dir):
        print(f"Validation directory not found: {val_dir}")
        exit(1)
    
    # Get images with ground truth labels
    all_val_images = [os.path.join(val_dir, f) for f in os.listdir(val_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Filter to only include images with corresponding non-empty label files
    labeled_images = []
    for img_path in all_val_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join("./yolo_dataset/labels/val", f"{base_name}.txt")
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            labeled_images.append(img_path)
    
    if not labeled_images:
        print("No validation images with label files found")
        exit(1)
    
    # Select random images with labels
    test_images = random.sample(labeled_images, min(NUM_TEST_IMAGES, len(labeled_images)))
    
    # Process each test image
    print(f"Found {len(test_images)} validation images with ground truth labels")
    
    for i, test_image in enumerate(test_images):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(test_images)}] Testing image: {test_image}")
        print(f"{'='*80}")
        try:
            results = detect_traffic_lights(model, test_image)
            if results["ground_truth"] is None:
                print("⚠️ Warning: No ground truth found despite earlier check")
        except Exception as e:
            print(f"Error during detection: {e}")