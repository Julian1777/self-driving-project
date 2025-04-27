import cv2 as cv

def visualize_annotations(image_path, anno_path):
    """
    Draw lane annotations on an image to visually verify they match.
    
    Args:
        image_path: Path to the image file
        anno_path: Path to the annotation file with coordinates
    """
    # Read the image
    img = cv.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Read the annotation file
    with open(anno_path, 'r') as f:
        lines = f.readlines()
    
    # Create a copy of the image for drawing
    visualization = img.copy()
    
    # Define colors for different lanes
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255)   # Magenta
    ]
    
    # Process each line of coordinates (each line is a separate lane)
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        # Extract coordinate pairs from the line
        coordinates = line.strip().split()
        points = []
        
        # Convert string coordinates to integer point pairs
        for j in range(0, len(coordinates), 2):
            if j+1 < len(coordinates):
                try:
                    x = int(float(coordinates[j]))
                    y = int(float(coordinates[j+1]))
                    points.append((x, y))
                except (ValueError, IndexError):
                    print(f"Warning: Invalid coordinate pair at line {i+1}, position {j}")
        
        # Draw the lane line
        color = colors[i % len(colors)]
        for k in range(len(points) - 1):
            cv.line(visualization, points[k], points[k+1], color, 3)
        
        # Draw points at each coordinate
        for point in points:
            cv.circle(visualization, point, 5, color, -1)
    
    # Resize if image is too large
    height, width = visualization.shape[:2]
    if width > 1200:
        scale = 1200 / width
        new_height = int(height * scale)
        visualization = cv.resize(visualization, (1200, new_height))
    
    # Display the result
    cv.imshow("Lane Annotations", visualization)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return visualization

ANNO_PATH = "/Users/jstamm2024/Documents/GitHub/learning-tf/lane_detection_cnn/dataset/culane/annotations/img_1_anno.txt"
IMG_PATH = "/Users/jstamm2024/Documents/GitHub/learning-tf/lane_detection_cnn/dataset/culane/images/img_1.jpg"
print("Verifying lane annotations...")
visualize_annotations(IMG_PATH, ANNO_PATH)