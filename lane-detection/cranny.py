import cv2 as cv
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv.fillPoly(mask, vertices, (255, 255, 255))
    masked_image = cv.bitwise_and(img, mask)
    return masked_image


image = cv.imread("flower.jpg", cv.IMREAD_GRAYSCALE)

canny_image = cv.Canny(image, threshold1=130, threshold2=160)

"""
cv.namedWindow('canny_image')
cv.moveWindow('canny_image', 0,0)
image = cv.resize(canny_image, (510, 510))
cv.imshow('canny_image', image)
"""

image_lane = cv.imread("lane.jpeg", cv.IMREAD_COLOR)

hsv_image = cv.cvtColor(image_lane, cv.COLOR_BGR2HSV)

# Range for white lanes
lower_white = np.array([0, 0, 200])
upper_white = np.array([255, 30, 255])

# Range for yellow lanes
lower_yellow = np.array([15, 100, 100])
upper_yellow = np.array([35, 255, 255])


white_mask = cv.inRange(hsv_image, lower_white, upper_white)
yellow_mask = cv.inRange(hsv_image, lower_yellow, upper_yellow)
combined_mask = cv.bitwise_or(white_mask, yellow_mask)

combined_mask = cv.resize(combined_mask, (image_lane.shape[1], image_lane.shape[0]))

mask_colored = cv.cvtColor(combined_mask, cv.COLOR_GRAY2BGR)

cv.imshow('Combined Mask', mask_colored)

filtered_image = cv.bitwise_and(image_lane, mask_colored)

vertices = np.array([[
    (100, 540),
    (900, 540),
    (550, 300),
    (400, 300),
]], dtype=np.int32)

roi_image = image_lane.copy()

cv.rectangle(roi_image, (400, 300), (900, 540), (0, 0, 255), 3)

roi_filtered_image = region_of_interest(filtered_image, vertices)

roi_image_resized = cv.resize(roi_image, (510, 510))
roi_filtered_image_resized = cv.resize(roi_filtered_image, (510, 510))

cv.imshow('ROI on Original Image', roi_image_resized)
cv.imshow('Filtered Image with ROI', roi_filtered_image_resized)

cv.waitKey(0)
cv.destroyAllWindows()