import cv2 as cv
import numpy as np

# Load your image
image = cv.imread("highway.jpg")
height, width = image.shape[:2]

# Create a copy for visualization
visualization = image.copy()

# Draw coordinate grid
grid_spacing = 100
for x in range(0, width, grid_spacing):
    # Draw vertical lines
    cv.line(visualization, (x, 0), (x, height), (255, 255, 255), 1)
    # Label x coordinate
    cv.putText(visualization, f"{x}", (x+5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Label percentage
    percentage = int(100 * x / width)
    cv.putText(visualization, f"{percentage}%", (x+5, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

for y in range(0, height, grid_spacing):
    # Draw horizontal lines
    cv.line(visualization, (0, y), (width, y), (255, 255, 255), 1)
    # Label y coordinate
    cv.putText(visualization, f"{y}", (5, y+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Label percentage
    percentage = int(100 * y / height)
    cv.putText(visualization, f"{percentage}%", (40, y+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Now draw the ROI polygon with different example values
def draw_roi(img, points, color, label):
    """
    Draw a polygon ROI defined by a list of points given as (x%, y%)
    """
    # Convert percentage points to pixel coordinates
    pixel_points = []
    for point in points:
        x_pct, y_pct = point
        x = int(x_pct * width)
        y = int(y_pct * height)
        pixel_points.append((x, y))
    
    # Create the polygon
    polygon = np.array([pixel_points], dtype=np.int32)
    
    # Draw the polygon
    cv.polylines(img, polygon, True, color, 2)
    
    # Add the label near the top of the polygon
    top_y = min([p[1] for p in pixel_points])
    top_x = pixel_points[[p[1] for p in pixel_points].index(top_y)][0]
    cv.putText(img, label, (top_x, top_y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Label each point
    point_labels = ["BL", "ML1", "TL", "TR", "MR1", "BR"]  # Add more labels as needed
    for i, (x, y) in enumerate(pixel_points):
        # Draw point
        cv.circle(img, (x, y), 5, color, -1)
        
        # Create label with index and percentage
        if i < len(point_labels):
            label = f"{point_labels[i]}: ({points[i][0]*100}%, {points[i][1]*100}%)"
        else:
            label = f"P{i}: ({points[i][0]*100}%, {points[i][1]*100}%)"
            
        # Position label based on location in image
        if y < height/2:  # Top half of the image
            label_y = y + 20
        else:  # Bottom half of the image
            label_y = y - 20
            
        cv.putText(img, label, (x+10, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Draw your current ROI
current_roi = visualization.copy()
draw_roi(current_roi, [
         (0.0, 1.0),           # Bottom left (0%, 100%)
         (0.15, 0.65), #mid left
         (0.4, 0.45),          # Top left (35%, 30%)
         (0.6, 0.45),          # Top right (65%, 30%)
         (0.95, 0.7), #mid right
         (1.0, 1.0)           # Bottom right (100%, 100%)
], (0, 255, 0), "Current ROI")

# Draw a wider ROI example
wider_roi = visualization.copy()
draw_roi(wider_roi, [
    (0.0, 1.0),           # Bottom left
    (0.2, 0.6),           # Top left
    (0.8, 0.6),           # Top right
    (1.0, 1.0)            # Bottom right
], (0, 255, 255), "Wider ROI")

# Draw a narrower ROI example
narrower_roi = visualization.copy()
draw_roi(narrower_roi, [
    (0.3, 1.0),           # Bottom left
    (0.4, 0.5),           # Top left
    (0.6, 0.5),           # Top right
    (0.7, 1.0)            # Bottom right
], (255, 0, 0), "Narrower ROI")

# Show the visualizations
cv.namedWindow('Grid & Coordinates')
cv.namedWindow('Current ROI')
cv.namedWindow('Wider ROI')
cv.namedWindow('Narrower ROI')

cv.moveWindow('Grid & Coordinates', 0, 0)
cv.moveWindow('Current ROI', 600, 0)
cv.moveWindow('Wider ROI', 0, 600)
cv.moveWindow('Narrower ROI', 600, 600)

grid_resized = cv.resize(visualization, (550, 400))
current_resized = cv.resize(current_roi, (550, 400))
wider_resized = cv.resize(wider_roi, (550, 400))
narrower_resized = cv.resize(narrower_roi, (550, 400))

cv.imshow('Grid & Coordinates', grid_resized)
cv.imshow('Current ROI', current_resized)
cv.imshow('Wider ROI', wider_resized)
cv.imshow('Narrower ROI', narrower_resized)

cv.waitKey(0)
cv.destroyAllWindows()