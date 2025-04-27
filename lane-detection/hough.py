import cv2 as cv
import numpy as np

lane_color = cv.imread("lane.jpeg")
height, width = lane_color.shape[:2]

gray = cv.cvtColor(lane_color, cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray, threshold1=140, threshold2=150)

cv.namedWindow('Edges')
cv.moveWindow('Edges', 0, 0)
edges_resized = cv.resize(edges, (510, 510))
cv.imshow('Edges', edges_resized)

lines = cv.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi/180,
    threshold=80,
    minLineLength=100,
    maxLineGap=50
)

all_lines_image = lane_color.copy()

if lines is not None:
    for line in lines:
        print(f"Hough detected {len(lines) if lines is not None else 0} lines")
        x1, y1, x2, y2 = line[0]
        cv.line(all_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

roi_left = width * 0.2
roi_right = width * 0.8

left_candidates = []
right_candidates = []

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if x2 == x1:
            continue
            
        slope = (y2 - y1) / (x2 - x1)
        
        if abs(slope) < 0.15 or abs(slope) > 10.0:
            continue
        
        bottom_y = max(y1, y2)
        bottom_x = x1 if y1 > y2 else x2
        print(f"Left candidates: {len(left_candidates)}")
        print(f"Right candidates: {len(right_candidates)}")
        mid_x = (x1 + x2) / 2

        if slope < 0 and width*0.2 < bottom_x < width*0.55:
            left_candidates.append(line)
        elif slope > 0 and width*0.45 < bottom_x < width*0.8:
            right_candidates.append(line)

        

current_lane_image = lane_color.copy()

best_left_line = None
min_left_x = 0

for line in left_candidates:
    x1, y1, x2, y2 = line[0]
    bottom_x = x1 if y1 > y2 else x2
    if bottom_x > min_left_x:
        min_left_x = bottom_x
        best_left_line = line

best_right_line = None
min_right_x = width

for line in right_candidates:
    x1, y1, x2, y2 = line[0]
    bottom_x = x1 if y1 > y2 else x2
    if bottom_x < min_right_x:
        min_right_x = bottom_x
        best_right_line = line

if best_left_line is not None:
    x1, y1, x2, y2 = best_left_line[0]
    cv.line(current_lane_image, (x1, y1), (x2, y2), (255, 0, 0), 4)  # Left boundary in blue

if best_right_line is not None:
    x1, y1, x2, y2 = best_right_line[0]
    cv.line(current_lane_image, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Right boundary in green


if lines is not None:
    rejected_by_slope = 0
    rejected_by_position = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if x2 == x1:
            continue
            
        slope = (y2 - y1) / (x2 - x1)
        
        if abs(slope) < 0.3 or abs(slope) > 5.0:
            rejected_by_slope += 1
            continue
        
        bottom_y = max(y1, y2)
        bottom_x = x1 if y1 > y2 else x2
        
        # Check if the line would be rejected by position
        if slope < 0:
            if not (width*0.3 < bottom_x < width*0.5):
                rejected_by_position += 1
        else:  # slope > 0
            if not (width*0.5 < bottom_x < width*0.7):
                rejected_by_position += 1
    
    print(f"Rejected by slope: {rejected_by_slope}")
    print(f"Rejected by position: {rejected_by_position}")

cv.namedWindow('All Detected Lines')
cv.namedWindow('Your Lane Boundaries')
cv.moveWindow('All Detected Lines', 520, 0)
cv.moveWindow('Your Lane Boundaries', 1040, 0)

all_lines_resized = cv.resize(all_lines_image, (510, 510))
current_lane_resized = cv.resize(current_lane_image, (510, 510))

cv.imshow('All Detected Lines', all_lines_resized)
cv.imshow('Your Lane Boundaries', current_lane_resized)

cv.waitKey(0)
cv.destroyAllWindows()