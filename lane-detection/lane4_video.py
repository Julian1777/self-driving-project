import cv2 as cv
import numpy as np



def canny(image):
    gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 50, 150)
    return canny

def roi(image):
    height, width = image.shape[:2]
    mask = np.zeros_like(image)
    polygon = np.array([[
        (0, height),           # Bottom left
        (0, height*0.8),     # Mid-left point
        (width*0.37, height*0.5),      # Top left
        (width*0.57, height*0.5),      # Top right
        (width, height*0.76),      # Mid-right point
        (width, height)            # Bottom right
    ]], dtype=np.int32)

    cv.fillPoly(mask, polygon, 255)
    masked_img = cv.bitwise_and(image, mask)
    return masked_img, polygon

def filter_lanes(lines, image_width):
    if lines is None:
        return None, None, None
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        mid_x = (x1 + x2) / 2

        if slope < 0 and mid_x > image_width * 0.5 and mid_x < image_width * 0.5:
            left_lines.append(line)
        elif slope > 0 and mid_x > image_width * 0.7:
            right_lines.append(line) 

    return left_lines, right_lines

def make_coordinates(image,line_parameters):
    slope,intercept=line_parameters
    y1=image.shape[0]
    y2=int(y1*0.6)
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

prev_right_fit_average = np.array([-0.5, 300])
prev_left_fit_average = np.array([0.5, -50])

def fit_lane(image, lines):
    global prev_left_fit_average, prev_right_fit_average

    if lines is None:
        left_line = make_coordinates(image, prev_left_fit_average)
        right_line = make_coordinates(image, prev_right_fit_average)
        return np.array([left_line, right_line])

    left_fit = []
    right_fit = []
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if abs(x2 - x1) < 1 or abs(y2 - y1) < 1:
            continue
            
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        if slope < -0.3:  # Left lane
            left_fit.append((slope, intercept))
        elif slope > 0.3:  # Right lane
            right_fit.append((slope, intercept))

    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        prev_left_fit_average = left_fit_average
        left_fit_average = prev_left_fit_average
        
    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        prev_right_fit_average = right_fit_average
    else:
        right_fit_average = prev_right_fit_average
        
    try:
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
    except Exception as e:
        print(f"Error calculating lines: {e}")
        left_line = make_coordinates(image, prev_left_fit_average)
        right_line = make_coordinates(image, prev_right_fit_average)
        
    return np.array([left_line, right_line])

def create_lane_path(image, lane_lines):
    lane_path = np.zeros_like(image)
    if lane_lines is None or len(lane_lines) < 2:
        return lane_path
    
    left_line = lane_lines[0]
    right_line = lane_lines[1]
    
    lane_points = np.array([
        [left_line[0], left_line[1]],   # Bottom left
        [left_line[2], left_line[3]],   # Top left
        [right_line[2], right_line[3]], # Top right
        [right_line[0], right_line[1]]  # Bottom right
    ], dtype=np.int32)
    
    cv.fillPoly(lane_path, [lane_points], (0, 100, 0))
    
    return lane_path
 
def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),8)

    return line_image

cap = cv.VideoCapture('test_video_2.mp4')
while(cap.isOpened()):
    ret ,frame=cap.read()

    height, width = frame.shape[:2]

    canny_image=canny(frame)
    
    cropped_image, roi_polygon=roi(canny_image)
    
    lines = cv.HoughLinesP(
        cropped_image,
        rho=2,
        theta=np.pi/180,
        threshold=100,
        minLineLength=40,
        maxLineGap=5
    )

    left_lines, right_lines = filter_lanes(lines, width)
    
    left_right_lanes = fit_lane(frame, lines)
    center_lane = None

    line_image = np.zeros_like(frame)

    lane_path = create_lane_path(frame, left_right_lanes)

    if left_right_lanes is not None and len(left_right_lanes) > 0:
        x1, y1, x2, y2 = left_right_lanes[0]
        cv.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 8)  # Green
    
    if left_right_lanes is not None and len(left_right_lanes) > 1:
        x1, y1, x2, y2 = left_right_lanes[1]
        cv.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 8)  # Red
    
    result = frame.copy()
    path_overlay = cv.addWeighted(result, 1, lane_path, 0.3, 0)  # 30% opacity for the path
    final_image = cv.addWeighted(path_overlay, 1, line_image, 1, 0)  # Add lines at full opacity
    
    cv.imshow('result', final_image)
    roi_resized = cv.resize(cropped_image, (400, 400))
    cv.imshow("ROI-Image", roi_resized)

    if cv.waitKey(1) & 0xFF==27:
        break
cap.release()
cv.destroyAllWindows()