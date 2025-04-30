import cv2
import numpy as np
from lane_detection_hough import lane_detection
from lane_detection_model import predict_lane
from sign_detection import predict_sign
from traffic_light_detect import detect_traffic_light
from traffic_light_class import predict_traffic_light

cap = cv2.VideoCapture("test_video_ams_cut.mp4")
ret, frame = cap.read()
cap.release()
assert ret, "Could not read frame"

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print("Processing started...")

hough = lane_detection(rgb)
print("Hough lines complete")

ml = predict_lane(rgb)
print("ML lanes complete")

signs = predict_sign(rgb)
print("Sign detection complete")

tdet = detect_traffic_light(rgb)
print("Traffic light detection complete")

tcls = predict_traffic_light(rgb)
print("Traffic light classification complete")

print("\nResults:")
print("Hough lines:", hough)
print("ML lines:", ml)
print("Signs:", signs)
print("TL detect:", tdet)
print("TL classify:", tcls)

vis = frame.copy()

if isinstance(hough, (list, tuple, np.ndarray)) and len(hough) > 0:
    for line in hough:
        try:
            if isinstance(line, tuple) and len(line) == 2:
                pt1, pt2 = line
                # Ensure points are int tuples
                pt1 = tuple(map(int, pt1))
                pt2 = tuple(map(int, pt2))
                cv2.line(vis, pt1, pt2, (0,0,255), 2)
        except Exception as e:
            print(f"Error drawing Hough line: {e}")

if isinstance(ml, (list, tuple, np.ndarray)) and len(ml) > 0:
    for line in ml:
        try:
            if isinstance(line, tuple) and len(line) == 2:
                pt1, pt2 = line
                # Ensure points are int tuples
                pt1 = tuple(map(int, pt1))
                pt2 = tuple(map(int, pt2))
                cv2.line(vis, pt1, pt2, (0,255,0), 2)
        except Exception as e:
            print(f"Error drawing ML line: {e}")

# Draw sign boxes
for sign in signs:
    if 'bbox' in sign:
        x, y, w, h = sign['bbox']
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255,0,0), 2)
        if 'label' in sign:
            cv2.putText(vis, f"{sign['label']} {sign['confidence']:.2f}", 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

# Draw traffic light boxes
for light in tdet:
    if 'bbox' in light:
        x1, y1, x2, y2 = light['bbox']
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,255), 2)
        if 'confidence' in light:
            cv2.putText(vis, f"TL: {light['confidence']:.2f}", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

# Draw traffic light states
for light in tcls:
    if 'state' in light:
        cv2.putText(vis, f"{light['state']} {light['confidence']:.2f}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

# Show the result
vis = cv2.resize(vis, (960, 540))
cv2.imshow("combined debug", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()