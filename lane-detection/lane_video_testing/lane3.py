import cv2 as cv
import numpy as np


def roi(img):
    mask = np.zeros_like(img)
    polygon = np.array([[
        (0, height),           # Bottom left
        (0, height*0.8),     # Mid-left point
        (width*0.47, height*0.5),      # Top left
        (width*0.57, height*0.5),      # Top right
        (width, height*0.76),      # Mid-right point
        (width, height)            # Bottom right
    ]], dtype=np.int32)

    cv.fillPoly(mask, polygon, 255)
    masked_img = cv.bitwise_and(img, mask)
    return masked_img, polygon

def filter_lanes(lines, image_width):
    if lines is None:
        return None
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2==x1:
            continue

        slope = (y2 - y1) / (x2 - x1)

        mid_x = (x1 + x2) / 2

        if slope < 0 and mid_x < image_width * 0.7:
            left_lines.append(line)
        elif slope > 0 and mid_x > image_width * 0.3:
            right_lines.append(line) 

    return left_lines, right_lines

def fit_lane(lines, img_height):
    if lines is None or len(lines) == 0:
        return None
    x_points = []
    y_points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_points.extend([x1, x2])
        y_points.extend([y1, y2])

    if len(x_points) > 2:
        median_x = np.median(x_points)
        std_x = np.std(x_points)
        
        filtered_indices = [i for i in range(len(x_points)) 
                            if abs(x_points[i] - median_x) < 2.5 * std_x]
        
        x_points = [x_points[i] for i in filtered_indices]
        y_points = [y_points[i] for i in filtered_indices]
        
        if len(x_points) < 2:
            print("Not enough points after filtering")
            return None

    try:
        coeffs = np.polyfit(y_points, x_points, 1)

        y_values = np.linspace(min(y_points), img_height, 10)
        x_values = coeffs[0] * y_values + coeffs[1]

        if len(y_points) > 5:
            coeffs2 = np.polyfit(y_points, x_points, 2)
            x_values2 = coeffs2[0] * y_values**2 + coeffs2[1] * y_values + coeffs2[2]
            max_deviation = max(abs(x_values2 - x_values))
            curvature = abs(coeffs2[0])  # The 'a' coefficient dictates curvature

            if max_deviation< width *0.08 and curvature < 0.0001:
                x_values = x_values2
                print("Using quadratic fit")

        lane_points = np.column_stack((x_values.astype(int), y_values.astype(int)))
        return lane_points
    except:
        print("Error in fitting polynomial")
        return None

    

lane_color = cv.imread('lane3.jpg')
height, width = lane_color.shape[:2]

gray = cv.cvtColor(lane_color, cv.COLOR_BGR2GRAY)

blurred = cv.GaussianBlur(gray, (5, 5), 0)

edges = cv.Canny(blurred, 50, 150)


masked_edges, roi_polygon = roi(edges)

roi_with_outline = cv.cvtColor(masked_edges, cv.COLOR_GRAY2BGR)
cv.polylines(roi_with_outline, [roi_polygon], True, (0, 255, 0), 2)

#cv.namedWindow('Edges')
cv.namedWindow('ROI Edges')
#cv.moveWindow('Edges', 0, 0)
cv.moveWindow('ROI Edges', 520, 0)

edges_resized = cv.resize(edges, (510, 510))
roi_with_outline_resized = cv.resize(roi_with_outline, (510, 510))

#cv.imshow('Edges', edges_resized)
cv.imshow('ROI Edges', roi_with_outline_resized)


lines = cv.HoughLinesP(
    masked_edges,
    rho=1,
    theta=np.pi / 180,
    threshold=15,
    minLineLength=15,
    maxLineGap=20
)

all_lines_image = lane_color.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(all_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
else:
    print("No lines detected")

cv.namedWindow('All Detected Lines')
cv.moveWindow('All Detected Lines', 520, 520)
all_lines_resized = cv.resize(all_lines_image, (510, 510))
cv.imshow('All Detected Lines', all_lines_resized)


left_lines, right_lines = filter_lanes(lines, width)

left_lane_points = fit_lane(left_lines, height)
right_lane_points = fit_lane(right_lines, height)

lane_overlay = lane_color.copy()



if left_lane_points is not None:
    for i in range(len(left_lane_points)-1):
        cv.line(lane_overlay, 
                tuple(left_lane_points[i]), 
                tuple(left_lane_points[i+1]), 
                (0, 255, 0), 10)
    
    bottom_left = (max(0, left_lane_points[-1][0] - 50), height)
    bottom_right = (min(width, left_lane_points[-1][0] + 50), height)
    
    poly_points = np.vstack([left_lane_points, [bottom_right, bottom_left]])
    
    overlay = lane_overlay.copy()
    cv.fillPoly(overlay, [poly_points.astype(np.int32)], (0, 255, 0))
    
    lane_overlay = cv.addWeighted(overlay, 0.3, lane_overlay, 0.7, 0)

if right_lane_points is not None:
    for i in range(len(right_lane_points)-1):
        cv.line(lane_overlay, 
                tuple(right_lane_points[i]), 
                tuple(right_lane_points[i+1]), 
                (255, 0, 0), 10)
    
    bottom_left = (max(0, right_lane_points[-1][0] - 50), height)
    bottom_right = (min(width, right_lane_points[-1][0] + 50), height)
    
    poly_points = np.vstack([right_lane_points, [bottom_right, bottom_left]])
    
    overlay = lane_overlay.copy()
    cv.fillPoly(overlay, [poly_points.astype(np.int32)], (0, 255, 0))  # Green
    lane_overlay = cv.addWeighted(overlay, 0.3, lane_overlay, 0.7, 0)

cv.namedWindow('Lane Predictions')
cv.moveWindow('Lane Prediction', 0, 520)
prediction_resized = cv.resize(lane_overlay, (510, 510))
cv.imshow('Lane Predictions', prediction_resized)




cv.waitKey(0)
cv.destroyAllWindows()