"""
get vehicle center through like         
lane_center = (left_bottom + right_bottom) / 2.0
         
        if original_image_width is not None:
            vehicle_center = original_image_width / 2.0
        else:
            vehicle_center = binary_warped.shape[1] / 2.0

Vehicle position = center x of warped image

For each detected lane:
  lane_left_x = lane['left_boundary_x']
  lane_right_x = lane['right_boundary_x']
  
  if vehicle_x >= lane_left_x AND vehicle_x <= lane_right_x:
    return lane['lane_id']  # Found current lane!

If no match found:
  return None (vehicle not in detected lanes, edge case)
"""


def get_current_lane(lanes, vehicle_center=None, image_width=None, warped_width=None, debug=False):
    """
    Determine which lane vehicle is in and classify all lanes.
    
    Args:
        lanes: List of detected lane data
        vehicle_center: Vehicle center X position (in warped coordinates if warped_width provided)
        image_width: Original image width (fallback)
        warped_width: Warped image width for proper vehicle center calculation
        debug: Enable debug output
    
    Returns:
        Dict with current lane info and classified lanes
    """
    # Use warped image center if available, otherwise fall back to image_width/2
    if vehicle_center is None:
        if warped_width is not None:
            vehicle_center = warped_width / 2.0
        elif image_width is not None:
            vehicle_center = image_width / 2.0
        else:
            vehicle_center = 0.0
    
    if debug:
        print(f"[LANE_SELECT] vehicle_center={vehicle_center:.1f}px, warped_width={warped_width}, image_width={image_width}")
    
    if not hasattr(get_current_lane, 'previous_lane_id'):
        get_current_lane.previous_lane_id = None
    if not hasattr(get_current_lane, 'previous_lane_class'):
        get_current_lane.previous_lane_class = None
    
    current_lane = None
    classified_lanes = {}

    min_dist = float('inf')
    closest_lane = None
    
    for lane in lanes:
        left_boundary_x = lane['left_fitx'][-1]
        right_boundary_x = lane['right_fitx'][-1]
        lane_id = lane['lane_id']
        
        # Classify lane by position
        if lane_id == 0:
            lane_class = 'left'
        elif lane_id == 1:
            lane_class = 'center'
        elif lane_id == 2:
            lane_class = 'right'
        else:
            lane_class = f'lane_{lane_id}'
        
        lane_width = right_boundary_x - left_boundary_x
        lane_center_x = (left_boundary_x + right_boundary_x) / 2.0
        
        min_lane_width_px = 100  # Minimum reasonable lane width in warped pixels
        if lane_width < min_lane_width_px:
            if debug:
                print(f"[LANE_SELECT] Lane {lane_id} ({lane_class}) REJECTED: width={lane_width:.1f}px < {min_lane_width_px}px (noise)")
            continue
        
        classified_lanes[lane_class] = {
            'lane_id': lane_id,
            'lane_data': lane,
            'left_x': left_boundary_x,
            'right_x': right_boundary_x,
            'center_x': lane_center_x,
            'width': lane_width
        }

        if debug:
            print(f"[LANE_SELECT] Lane {lane_id} ({lane_class}): left={left_boundary_x:.1f}, right={right_boundary_x:.1f}, center={lane_center_x:.1f}, width={lane_width:.1f}")

        # check strict boundry inclusion
        if left_boundary_x <= vehicle_center <= right_boundary_x:
            position = (vehicle_center - left_boundary_x) / (right_boundary_x - left_boundary_x)
            current_lane = {
                'lane_id': lane_id,
                'lane_class': lane_class,
                'position_in_lane': position,
                'lane_data': lane
            }
            if debug:
                print(f"[LANE_SELECT] {lane_id} ({lane_class}), position={position:.2f}")
        # Keep track of closest lane as fallback
        dist = abs(vehicle_center - lane_center_x)
        if dist < min_dist:
            min_dist = dist
            closest_lane = {
                'lane_id': lane_id,
                'lane_class': lane_class,
                'position_in_lane': 0.5, # approx center
                'lane_data': lane
            }
            if debug:
                print(f"[LANE_SELECT] Closest so far: lane {lane_id} ({lane_class}), dist={dist:.1f}px")
            
        # if outside of bounds return to closest lane
        if current_lane is None and closest_lane is not None:
            current_lane = closest_lane
            if debug:
                print(f"[LANE_SELECT] Fallback to closest: lane {closest_lane['lane_id']} ({closest_lane['lane_class']})")
    
    if current_lane and get_current_lane.previous_lane_id is not None:
        prev_lane_id = get_current_lane.previous_lane_id
        prev_lane_class = get_current_lane.previous_lane_class
        if prev_lane_class in classified_lanes:
            prev_lane_info = classified_lanes[prev_lane_class]
            prev_left = prev_lane_info['left_x']
            prev_right = prev_lane_info['right_x']
            margin = 20  # pixels
            if prev_left - margin <= vehicle_center <= prev_right + margin:
                if debug:
                    print(f"[LANE_SELECT] HYSTERESIS: Keeping previous lane {prev_lane_id} ({prev_lane_class})")
                current_lane = {
                    'lane_id': prev_lane_id,
                    'lane_class': prev_lane_class,
                    'position_in_lane': (vehicle_center - prev_left) / (prev_right - prev_left),
                    'lane_data': prev_lane_info['lane_data']
                }
    
    if current_lane:
        get_current_lane.previous_lane_id = current_lane['lane_id']
        get_current_lane.previous_lane_class = current_lane['lane_class']
    
    if debug:
        if current_lane:
            print(f"[LANE_SELECT] FINAL: Selected lane {current_lane['lane_id']} ({current_lane['lane_class']}), position_in_lane={current_lane['position_in_lane']:.2f}")
        else:
            print(f"[LANE_SELECT] FINAL: No Lane Selected")
    
    return {
        'current_lane': current_lane,
        'all_lanes': classified_lanes
    }
