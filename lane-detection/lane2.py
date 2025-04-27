import cv2 as cv
import numpy as np


#Function to set the region of interest on the image
#Opencv uses left to right 0 - 100% and left top to bottom left 0-100%
def region_of_interest(img):
    mask = np.zeros_like(img)
    polygon = np.array([[
        (0, height),           # Bottom left
        (width*0.13, height*0.72),     # Mid-left point
        (width*0.4, height*0.47),      # Top left
        (width*0.6, height*0.47),      # Top right
        (width*0.92, height*0.75),      # Mid-right point
        (width, height)            # Bottom right
    ]], dtype=np.int32)
    #Fill the polygon created above
    cv.fillPoly(mask, polygon, 255)
    #Apply the mask (ROI) to the image
    masked_image = cv.bitwise_and(img, mask)
    return masked_image, polygon


def average_lane_line(lines, height, width, is_left=True, degree=1):
    """
    Fits a polynomial to lane lines and returns points for drawing
    
    Args:
        lines: List of detected line segments
        height: Image height
        width: Image width
        is_left: Whether this is the left lane (True) or right lane (False)
        degree: Polynomial degree (1=linear, 2=quadratic for curves)
        
    Returns:
        List of points to draw the fitted lane line
    """
    if len(lines) == 0:
        return None
    
    # Get all points from the lines
    x_coords = []
    y_coords = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])

    # Weight points by their y-position (closer to bottom = higher weight)
    weights = [y/height for y in y_coords]  # Normalize to 0-1
    
    # Fit a line to all points using polyfit
    if len(x_coords) > degree:  # Need more points than degree
        try:
            coeffs = np.polyfit(y_coords, x_coords, degree, w=weights)
            
            # For straight lines (degree=1), ensure the fit is reasonable
            if degree == 1:
                slope, intercept = coeffs
                # Check if slope is reasonable for a lane line
                if is_left and slope > 0:  # Left lines should have negative slope
                    print("Warning: Left lane has positive slope, might be incorrect fit")
                elif not is_left and slope < 0:  # Right lines should have positive slope
                    print("Warning: Right lane has negative slope, might be incorrect fit")
            
            # Generate points along the polynomial at regular y intervals
            y_points = np.linspace(height*0.6, height, num=20)
            x_points = np.polyval(coeffs, y_points)
            
            # Convert to integer pixel coordinates
            points = np.column_stack((x_points, y_points)).astype(np.int32)
            
            # Make sure points stay within image bounds
            points[:,0] = np.clip(points[:,0], 0, width-1)
            
            return points
        except np.linalg.LinAlgError:
            # Fallback to linear if polynomial fit fails
            try:
                coeffs = np.polyfit(y_coords, x_coords, 1, w=weights)
                slope, intercept = coeffs
                
                # Calculate points for top and bottom
                y_bottom = height
                y_top = height * 0.6
                x_bottom = int(slope * y_bottom + intercept)
                x_top = int(slope * y_top + intercept)
                
                return np.array([[x_bottom, int(y_bottom)], [x_top, int(y_top)]])
            except:
                return None
    return None


def draw_lane_lines(image, left_lanes, right_lanes, fill_lane=True):
    """
    Draws multiple lane lines and fills the lane areas
    
    Args:
        image: The image to draw on
        left_lanes: List of points for each left lane line
        right_lanes: List of points for each right lane line
        fill_lane: Whether to fill the lane areas between lines
    
    Returns:
        Image with lane markings
    """
    result_image = image.copy()
    
    # Define colors for different lanes
    left_colors = [(255, 0, 0), (255, 0, 255), (255, 127, 0)]  # Blue, Magenta, Orange
    right_colors = [(0, 255, 0), (0, 255, 255), (0, 127, 255)]  # Green, Yellow, Light Blue
    
    # Draw left lane lines
    for i, points in enumerate(left_lanes):
        if len(points) >= 2:
            color = left_colors[i % len(left_colors)]
            for j in range(len(points) - 1):
                cv.line(result_image, 
                       tuple(points[j]), 
                       tuple(points[j+1]), 
                       color, 8)  # Thick line
    
    # Draw right lane lines
    for i, points in enumerate(right_lanes):
        if len(points) >= 2:
            color = right_colors[i % len(right_colors)]
            for j in range(len(points) - 1):
                cv.line(result_image, 
                       tuple(points[j]), 
                       tuple(points[j+1]), 
                       color, 8)  # Thick line
    
    # Fill lane areas if requested
    if fill_lane:
        # Fill the areas between consecutive lane markings
        lane_fill_colors = [
            (180, 180, 0),    # Light blue
            (0, 180, 180),    # Yellow
            (180, 0, 180)     # Magenta
        ]
        
        # Fill lanes between left markings
        for i in range(len(left_lanes) - 1):
            mask = np.zeros_like(image)
            lane_pts = np.vstack((left_lanes[i], np.flipud(left_lanes[i+1])))
            color_idx = i % len(lane_fill_colors)
            cv.fillPoly(mask, [lane_pts], lane_fill_colors[color_idx])
            result_image = cv.addWeighted(result_image, 1, mask, 0.2, 0)
        
        # Fill lanes between right markings
        for i in range(len(right_lanes) - 1):
            mask = np.zeros_like(image)
            lane_pts = np.vstack((right_lanes[i], np.flipud(right_lanes[i+1])))
            color_idx = i % len(lane_fill_colors)
            cv.fillPoly(mask, [lane_pts], lane_fill_colors[color_idx])
            result_image = cv.addWeighted(result_image, 1, mask, 0.2, 0)
        
        # Fill center lane if we have both left and right lanes
        if len(left_lanes) > 0 and len(right_lanes) > 0:
            mask = np.zeros_like(image)
            center_lane_pts = np.vstack((left_lanes[-1], np.flipud(right_lanes[0])))
            cv.fillPoly(mask, [center_lane_pts], (0, 0, 180))  # Red for center lane
            result_image = cv.addWeighted(result_image, 1, mask, 0.2, 0)
    
    return result_image


def remove_outliers(lines, height, width, is_left=True):
    """
    Removes outlier line segments that don't match the general trend
    Args:
        lines: List of line segments
        height: Image height
        width: Image width
        is_left: Whether to filter left or right lines
    """
    if len(lines) <= 2:  # Need at least 3 lines for meaningful filtering
        return lines
    
    # Get slopes and positions of all lines
    slopes = []
    midpoints = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Skip vertical lines
        if x2 == x1:
            continue
            
        # Calculate slope and midpoint
        slope = (y2 - y1) / (x2 - x1)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        slopes.append(slope)
        midpoints.append((mid_x, mid_y))
    
    # Calculate median slope and standard deviation
    median_slope = np.median(slopes)
    slope_std = np.std(slopes)
    
    # Set threshold for outlier removal
    slope_threshold = max(0.4, slope_std * 2.5)  # At least 0.4 or 2.5 standard deviations
    
    # Filter lines based on slope difference from median
    filtered_lines = []
    for i, line in enumerate(lines):
        if i >= len(slopes):  # Skip vertical lines that were excluded
            continue
            
        slope_diff = abs(slopes[i] - median_slope)
        
        # Keep lines with similar slopes to median
        if slope_diff <= slope_threshold:
            filtered_lines.append(line)
    
    # Make sure we keep at least a minimum number of lines
    if len(filtered_lines) < 2 and len(lines) > 0:
        # Sort by slope difference and keep the closest 2
        sorted_indices = np.argsort([abs(slopes[i] - median_slope) for i in range(len(slopes))])
        filtered_lines = [lines[i] for i in sorted_indices[:2] if i < len(lines)]
    
    return filtered_lines


def group_lanes_by_position(lines, height, width, is_left=True):
    """
    Group lane lines into separate lanes based on their horizontal position
    
    Args:
        lines: List of detected line segments
        height: Image height
        width: Image width
        is_left: Whether these are left side lines (True) or right side (False)
    
    Returns:
        List of line groups, each group representing one lane
    """
    if len(lines) <= 1:
        return [lines] if lines else []
    
    # Calculate x-position at bottom of image for each line
    bottom_x_positions = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if x2 == x1:  # Skip vertical lines
            continue
            
        # Calculate slope and intercept
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        # Calculate x-position at bottom of image (y = height)
        bottom_x = int((height - intercept) / slope) if slope != 0 else x1
        
        bottom_x_positions.append((bottom_x, line))
    
    # Sort by x-position
    bottom_x_positions.sort(key=lambda x: x[0])
    
    # Initialize groups
    lane_groups = []
    current_group = []
    
    # If no positions calculated, return empty list
    if not bottom_x_positions:
        return []
    
    # Start first group
    current_group = [bottom_x_positions[0][1]]
    prev_x = bottom_x_positions[0][0]
    
    # Group by x-position with a dynamic threshold
    # Wider spacing for positions further from camera
    lane_width_threshold = width * 0.05  # 5% of image width minimum
    
    for i in range(1, len(bottom_x_positions)):
        current_x, line = bottom_x_positions[i]
        x_diff = abs(current_x - prev_x)
        
        # If this line is close to previous, add to current group
        if x_diff < lane_width_threshold:
            current_group.append(line)
        # Otherwise start a new group
        else:
            lane_groups.append(current_group)
            current_group = [line]
        
        prev_x = current_x
    
    # Add the last group
    if current_group:
        lane_groups.append(current_group)
    
    # Filter out groups with too few lines
    lane_groups = [group for group in lane_groups if len(group) >= 1]
    
    print(f"Found {len(lane_groups)} {'left' if is_left else 'right'} lane groups")
    
    return lane_groups


def assess_road_curvature(lines, height, width):
    """
    Assess if the road is curved enough to warrant a higher degree polynomial
    
    Returns:
        True if road is curved, False if straight
    """
    if len(lines) < 4:
        return False  # Not enough points to assess curvature
        
    # Get all points
    x_coords = []
    y_coords = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    
    # Try both linear and quadratic fit
    try:
        # Linear fit
        linear_coeffs = np.polyfit(y_coords, x_coords, 1)
        linear_residuals = np.sum((np.polyval(linear_coeffs, y_coords) - x_coords)**2)
        
        # Quadratic fit
        quad_coeffs = np.polyfit(y_coords, x_coords, 2)
        quad_residuals = np.sum((np.polyval(quad_coeffs, y_coords) - x_coords)**2)
        
        # Calculate improvement from quadratic
        if linear_residuals > 0:
            improvement = (linear_residuals - quad_residuals) / linear_residuals
            
            # If quadratic improves fit by more than 20%, consider it curved
            return improvement > 0.2
            
    except:
        pass
        
    return False  # Default to straight if we can't determine


def sanity_check_lanes(lanes, width, height, is_left=True):
    """Check if the lane fits make sense"""
    valid_lanes = []
    
    for lane in lanes:
        valid = True
        
        if len(lane) < 2:
            continue
        
        # Check slope direction
        bottom_point = lane[-1]
        top_point = lane[0]
        if bottom_point[0] == top_point[0]:  # Avoid division by zero
            valid = False
        else:
            slope = (bottom_point[1] - top_point[1]) / (bottom_point[0] - top_point[0])
            
            # Left lanes should have negative slope, right lanes should have positive
            if (is_left and slope > 0) or (not is_left and slope < 0):
                valid = False
        
        # Check if lane is within reasonable bounds
        bottom_x = bottom_point[0]
        if is_left:
            if bottom_x < 0 or bottom_x > width * 0.6:
                valid = False
        else:
            if bottom_x < width * 0.4 or bottom_x > width:
                valid = False
        
        if valid:
            valid_lanes.append(lane)
    
    return valid_lanes


# Load the image and initialize
lane_color = cv.imread("highway.jpg")
height, width = lane_color.shape[:2]

# Convert to grayscale
gray = cv.cvtColor(lane_color, cv.COLOR_BGR2GRAY)

# Optional: Enhance contrast
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# Apply Gaussian blur
blurred = cv.GaussianBlur(enhanced, (5, 5), 0)

# Detect edges
edges = cv.Canny(blurred, threshold1=80, threshold2=170)

# Apply the ROI mask
masked_edges, roi_polygon = region_of_interest(edges)

# Visualize the ROI
roi_with_outline = cv.cvtColor(masked_edges, cv.COLOR_GRAY2BGR)
cv.polylines(roi_with_outline, [roi_polygon], True, (0, 255, 0), 2)

# Create windows for visualization
cv.namedWindow('Edges')
cv.namedWindow('ROI Edges')
cv.moveWindow('Edges', 0, 0)
cv.moveWindow('ROI Edges', 520, 0)

# Resize for display
edges_resized = cv.resize(edges, (510, 510))
roi_with_outline_resized = cv.resize(roi_with_outline, (510, 510))

# Show edge detection results
cv.imshow('Edges', edges_resized)
cv.imshow('ROI Edges', roi_with_outline_resized)

# Detect lines using Hough transform
lines = cv.HoughLinesP(
    masked_edges,
    rho=1,
    theta=np.pi/180,
    threshold=15,           # Lower threshold to detect fainter lines
    minLineLength=5,        # Detect shorter segments
    maxLineGap=10           # Don't merge separate lane markings
)

print(f"Hough detected {len(lines) if lines is not None else 0} lines")

# Show all detected lines
all_lines_image = lane_color.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(all_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv.namedWindow('All Detected Lines')
cv.moveWindow('All Detected Lines', 520, 520)
all_lines_resized = cv.resize(all_lines_image, (510, 510))
cv.imshow('All Detected Lines', all_lines_resized)

# Filter lines into left and right candidates
left_candidates = []
right_candidates = []

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Skip vertical lines (avoid division by zero)
        if x2 == x1:
            continue
            
        # Calculate line slope
        slope = (y2 - y1) / (x2 - x1)
        
        # Filter out nearly horizontal or too steep lines
        if abs(slope) < 0.2 or abs(slope) > 2.0:
            continue
        
        # Calculate line midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        center_region = 0.1 * width  # 10% buffer around center
        if width*0.5-center_region <= mid_x <= width*0.5+center_region:
            if abs(slope) >= 0.2:  # Already checking this above, but being explicit
                if slope < 0:
                    left_candidates.append(line)
                else:
                    right_candidates.append(line)
                
                print(f"Added center line at x={mid_x:.1f}, slope={slope:.2f}")
                continue
        
        # Lines with negative slope are left lane markers
        if slope < 0 and mid_x < width*0.65:
            left_candidates.append(line)
        # Lines with positive slope are right lane markers
        elif slope > 0 and mid_x > width*0.35:
            right_candidates.append(line)
    
    print(f"Left candidates: {len(left_candidates)}")
    print(f"Right candidates: {len(right_candidates)}")

# Create image to show line candidates
candidates_image = lane_color.copy()

# Draw left candidates in blue
for line in left_candidates:
    x1, y1, x2, y2 = line[0]
    cv.line(candidates_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue

# Draw right candidates in green
for line in right_candidates:
    x1, y1, x2, y2 = line[0]
    cv.line(candidates_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green

# Show candidates
cv.namedWindow('Line Candidates')
cv.moveWindow('Line Candidates', 1040, 0)
candidates_resized = cv.resize(candidates_image, (510, 510))
cv.imshow('Line Candidates', candidates_resized)

center_vis = lane_color.copy()

# Draw a rectangle showing center region
center_region_width = 0.1 * width
center_left = int(width*0.5 - center_region_width)
center_right = int(width*0.5 + center_region_width)
cv.rectangle(center_vis, 
            (center_left, int(height*0.6)), 
            (center_right, height),
            (255, 255, 255), 2)

# Find and highlight lines in the center region
center_left_lines = []  # Lines with negative slope in center
center_right_lines = []  # Lines with positive slope in center

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Skip vertical lines
        if x2 == x1:
            continue
            
        # Calculate slope and midpoint
        slope = (y2 - y1) / (x2 - x1)
        mid_x = (x1 + x2) / 2
        
        # Check if in center region
        if center_left <= mid_x <= center_right:
            if slope < 0:
                cv.line(center_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue
                center_left_lines.append(line)
            elif slope > 0:
                cv.line(center_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green
                center_right_lines.append(line)

# Add count of center lines to visualization
cv.putText(center_vis, f"Center lines (left slope): {len(center_left_lines)}", 
          (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
cv.putText(center_vis, f"Center lines (right slope): {len(center_right_lines)}", 
          (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Show center region analysis
cv.namedWindow('Center Line Analysis')
cv.moveWindow('Center Line Analysis', 0, 260)
center_resized = cv.resize(center_vis, (510, 510))
cv.imshow('Center Line Analysis', center_resized)

# Remove outliers from candidates
left_candidates = remove_outliers(left_candidates, height, width, is_left=True)
right_candidates = remove_outliers(right_candidates, height, width, is_left=False)

print(f"Left candidates after outlier removal: {len(left_candidates)}")
print(f"Right candidates after outlier removal: {len(right_candidates)}")

# Group lanes on each side
left_lane_groups = group_lanes_by_position(left_candidates, height, width, is_left=True)
right_lane_groups = group_lanes_by_position(right_candidates, height, width, is_left=False)

# Check if the road has curves
is_left_curved = assess_road_curvature(left_candidates, height, width)
is_right_curved = assess_road_curvature(right_candidates, height, width)

# Use appropriate polynomial degree based on curvature
left_degree = 2 if is_left_curved else 1
right_degree = 2 if is_right_curved else 1

print(f"Using degree {left_degree} for left lane (curved: {is_left_curved})")
print(f"Using degree {right_degree} for right lane (curved: {is_right_curved})")

# Process each lane group
left_lanes = []
right_lanes = []

for i, lane_group in enumerate(left_lane_groups):
    lane_points = average_lane_line(lane_group, height, width, is_left=True, degree=left_degree)
    if lane_points is not None:
        left_lanes.append(lane_points)
        print(f"Processed left lane {i+1} with {len(lane_group)} line segments")

for i, lane_group in enumerate(right_lane_groups):
    lane_points = average_lane_line(lane_group, height, width, is_left=False, degree=right_degree)
    if lane_points is not None:
        right_lanes.append(lane_points)
        print(f"Processed right lane {i+1} with {len(lane_group)} line segments")

# Sort lanes from left to right
if left_lanes:
    left_lanes.sort(key=lambda points: points[-1][0])  # Sort by bottom x-coordinate
if right_lanes:
    right_lanes.sort(key=lambda points: points[-1][0])  # Sort by bottom x-coordinate

# Run sanity checks on lane fits
left_lanes = sanity_check_lanes(left_lanes, width, height, is_left=True)
right_lanes = sanity_check_lanes(right_lanes, width, height, is_left=False)

print(f"Final lane count: {len(left_lanes)} left, {len(right_lanes)} right")

# Create a visualization showing all detected lanes
all_lanes_image = lane_color.copy()

# Define different colors for multiple lanes
left_colors = [(255, 0, 0), (255, 0, 255), (255, 127, 0)]  # Blue, Magenta, Orange
right_colors = [(0, 255, 0), (0, 255, 255), (0, 127, 255)]  # Green, Yellow, Light Blue

# Draw each left lane
for i, lane_points in enumerate(left_lanes):
    color = left_colors[i % len(left_colors)]
    for j in range(len(lane_points) - 1):
        cv.line(all_lanes_image, 
                tuple(lane_points[j]), 
                tuple(lane_points[j+1]), 
                color, 8)
    
    # Label each lane
    bottom_point = lane_points[-1]
    cv.putText(all_lanes_image, f"L{i+1}", 
               (bottom_point[0]+10, bottom_point[1]-10), 
               cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# Draw each right lane
for i, lane_points in enumerate(right_lanes):
    color = right_colors[i % len(right_colors)]
    for j in range(len(lane_points) - 1):
        cv.line(all_lanes_image, 
                tuple(lane_points[j]), 
                tuple(lane_points[j+1]), 
                color, 8)
    
    # Label each lane
    bottom_point = lane_points[-1]
    cv.putText(all_lanes_image, f"R{i+1}", 
               (bottom_point[0]-40, bottom_point[1]-10), 
               cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# Show all lanes
cv.namedWindow('All Lanes')
cv.moveWindow('All Lanes', 1040, 260)
all_lanes_resized = cv.resize(all_lanes_image, (510, 510))
cv.imshow('All Lanes', all_lanes_resized)

# Create visualization for fitted points
fitted_curves_image = lane_color.copy()

# Draw points for each left lane
for i, lane_points in enumerate(left_lanes):
    color = left_colors[i % len(left_colors)]
    for point in lane_points:
        cv.circle(fitted_curves_image, tuple(point), 5, color, -1)

# Draw points for each right lane
for i, lane_points in enumerate(right_lanes):
    color = right_colors[i % len(right_colors)]
    for point in lane_points:
        cv.circle(fitted_curves_image, tuple(point), 5, color, -1)

# Show the fitted curves visualization
cv.namedWindow('Fitted Curves')
cv.moveWindow('Fitted Curves', 1040, 520)
fitted_resized = cv.resize(fitted_curves_image, (510, 510))
cv.imshow('Fitted Curves', fitted_resized)

# Create enhanced visualization with lane markings and driving areas
enhanced_vis = lane_color.copy()

# Define lane fill colors with transparency
lane_fill_colors = [
    (180, 180, 0),    # Light blue
    (0, 180, 180),    # Yellow
    (180, 0, 180)     # Magenta
]

# Fill driving lanes between markings
if len(left_lanes) > 0 and len(right_lanes) > 0:
    # Start with leftmost lane (between road edge and first left lane marking)
    if len(left_lanes) > 0:
        leftmost_lane = left_lanes[0]
        road_edge = np.array([[0, height], [0, int(height*0.6)]])
        
        # Create left edge lane polygon
        edge_lane_pts = np.vstack((road_edge, np.flipud(leftmost_lane)))
        mask = np.zeros_like(enhanced_vis)
        cv.fillPoly(mask, [edge_lane_pts], lane_fill_colors[0])
        enhanced_vis = cv.addWeighted(enhanced_vis, 1, mask, 0.2, 0)
    
    # Fill lanes between left markings
    for i in range(len(left_lanes) - 1):
        mask = np.zeros_like(enhanced_vis)
        lane_pts = np.vstack((left_lanes[i], np.flipud(left_lanes[i+1])))
        color_idx = (i + 1) % len(lane_fill_colors)
        cv.fillPoly(mask, [lane_pts], lane_fill_colors[color_idx])
        enhanced_vis = cv.addWeighted(enhanced_vis, 1, mask, 0.2, 0)
    
    # Fill lanes between right markings
    for i in range(len(right_lanes) - 1):
        mask = np.zeros_like(enhanced_vis)
        lane_pts = np.vstack((right_lanes[i], np.flipud(right_lanes[i+1])))
        color_idx = (i + 1) % len(lane_fill_colors)
        cv.fillPoly(mask, [lane_pts], lane_fill_colors[color_idx])
        enhanced_vis = cv.addWeighted(enhanced_vis, 1, mask, 0.2, 0)
    
    # Fill rightmost lane (between last right marking and road edge)
    if len(right_lanes) > 0:
        rightmost_lane = right_lanes[-1]
        road_edge = np.array([[width, height], [width, int(height*0.6)]])
        
        # Create right edge lane polygon
        edge_lane_pts = np.vstack((rightmost_lane, np.flipud(road_edge)))
        mask = np.zeros_like(enhanced_vis)
        cv.fillPoly(mask, [edge_lane_pts], lane_fill_colors[0])
        enhanced_vis = cv.addWeighted(enhanced_vis, 1, mask, 0.2, 0)
    
    # Fill center lane (between rightmost left marking and leftmost right marking)
    if len(left_lanes) > 0 and len(right_lanes) > 0:
        center_lane_pts = np.vstack((left_lanes[-1], np.flipud(right_lanes[0])))
        mask = np.zeros_like(enhanced_vis)
        cv.fillPoly(mask, [center_lane_pts], (0, 0, 180))  # Red for center lane
        enhanced_vis = cv.addWeighted(enhanced_vis, 1, mask, 0.2, 0)

# Draw all lane markings on the enhanced visualization
for i, lane_points in enumerate(left_lanes):
    color = left_colors[i % len(left_colors)]
    for j in range(len(lane_points) - 1):
        cv.line(enhanced_vis, tuple(lane_points[j]), tuple(lane_points[j+1]), color, 4)

for i, lane_points in enumerate(right_lanes):
    color = right_colors[i % len(right_colors)]
    for j in range(len(lane_points) - 1):
        cv.line(enhanced_vis, tuple(lane_points[j]), tuple(lane_points[j+1]), color, 4)

# Calculate total number of lanes
total_lanes = len(left_lanes) + len(right_lanes)
if len(left_lanes) > 0 and len(right_lanes) > 0:
    total_lanes += 1  # Add center lane

# Add information about number of lanes
cv.putText(enhanced_vis, f"Total lanes: {total_lanes}", 
           (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv.putText(enhanced_vis, f"Left lanes: {len(left_lanes)}", 
           (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv.putText(enhanced_vis, f"Right lanes: {len(right_lanes)}", 
           (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Show the enhanced visualization
cv.namedWindow('Enhanced Lane Visualization')
cv.moveWindow('Enhanced Lane Visualization', 0, 520)
enhanced_resized = cv.resize(enhanced_vis, (510, 510))
cv.imshow('Enhanced Lane Visualization', enhanced_resized)

# Create a debug visualization showing all steps
debug_image = lane_color.copy()

# Draw the ROI polygon
cv.polylines(debug_image, [roi_polygon], True, (255, 255, 255), 1)

# Draw all detected lines faintly
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(debug_image, (x1, y1), (x2, y2), (100, 100, 100), 1)

# Draw candidate lines more visibly
for line in left_candidates:
    x1, y1, x2, y2 = line[0]
    cv.line(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
for line in right_candidates:
    x1, y1, x2, y2 = line[0]
    cv.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Draw fitted lanes with thicker lines
for i, points in enumerate(left_lanes):
    color = left_colors[i % len(left_colors)]
    for j in range(len(points) - 1):
        cv.line(debug_image, tuple(points[j]), tuple(points[j+1]), color, 3)
        
for i, points in enumerate(right_lanes):
    color = right_colors[i % len(right_colors)]
    for j in range(len(points) - 1):
        cv.line(debug_image, tuple(points[j]), tuple(points[j+1]), color, 3)
        
# Add explanation text
cv.putText(debug_image, "Gray: All lines", (10, 30), 
           cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
cv.putText(debug_image, "Blue: Left candidates", (10, 60), 
           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
cv.putText(debug_image, "Green: Right candidates", (10, 90), 
           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv.putText(debug_image, "Colored lines: Fitted lanes", (10, 120), 
           cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 200), 2)

# Show the debug visualization
cv.namedWindow('Lane Detection Debug')
cv.moveWindow('Lane Detection Debug', 0, 260)
debug_resized = cv.resize(debug_image, (510, 510))
cv.imshow('Lane Detection Debug', debug_resized)

# Wait for user input and close all windows when done
cv.waitKey(0)
cv.destroyAllWindows()