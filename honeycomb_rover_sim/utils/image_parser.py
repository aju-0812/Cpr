import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

def parse_arena_image(image_bytes) -> Dict[str, Any]:
    """
    Parses an arena image to extract obstacles, zones, and markers.
    Assumes standard color coding:
    - Blue circles: Obstacles
    - Orange circles: Craters (Obstacles)
    - Green rectangle: Start Zone
    - Red rectangle: Construction Zone
    - Markers: Not easily detectable from simple schematic without tags, 
               so we'll infer or use defaults if not found.
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image")

    # Resize to standard width for consistent processing (optional but good for scale)
    # Let's assume the image represents the 9x5m arena exactly.
    height, width, _ = img.shape
    scale_x = 9.0 / width
    scale_y = 5.0 / height
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    config = {
        'arena': {'width': 9.0, 'height': 5.0},
        'obstacles': [],
        'zones': {},
        'markers': [] # We might keep default markers or try to find them
    }
    
    # --- Detect Obstacles (Blue & Orange) ---
    # Blue mask
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Orange mask (Craters)
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Combine obstacles
    mask_obs = cv2.bitwise_or(mask_blue, mask_orange)
    
    contours, _ = cv2.findContours(mask_obs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # Get circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        
        # Convert to meters
        # Image Y is top-down, Arena Y is bottom-up? 
        # Usually images are (0,0) top-left. Arena (0,0) is bottom-left.
        # So y_arena = height_m - (y_img * scale_y)
        
        xm = x * scale_x
        ym = 5.0 - (y * scale_y)
        rm = radius * max(scale_x, scale_y)
        
        # Filter noise and huge false positives (e.g. arena border)
        if rm > 0.1 and rm < 1.5: 
            # Check if overlapping start pose (1.5, 1.0)
            # Distance to start
            dist_start = np.hypot(xm - 1.5, ym - 1.0)
            if dist_start < rm + 0.3: # 0.3 safety margin
                continue
                
            config['obstacles'].append({'x': float(xm), 'y': float(ym), 'r': float(rm)})

    # --- Detect Start Zone (Green) ---
    # Approximate green
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_green:
        # Assume largest green area is start
        cnt = max(contours_green, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        x_min = x * scale_x
        y_max = 5.0 - (y * scale_y)
        x_max = (x + w) * scale_x
        y_min = 5.0 - ((y + h) * scale_y)
        
        config['zones']['start'] = {
            'type': 'start',
            'rect': [float(x_min), float(y_min), float(x_max), float(y_max)]
        }

        # Filter obstacles inside start zone
        # This prevents legend/text boxes in the start area from being detected as obstacles
        filtered_obstacles = []
        for obs in config['obstacles']:
            ox, oy = obs['x'], obs['y']
            # Check if inside rect
            if not (x_min <= ox <= x_max and y_min <= oy <= y_max):
                filtered_obstacles.append(obs)
        config['obstacles'] = filtered_obstacles

    # --- Detect Construction Zone (Red) ---
    # Red wraps around 180 in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), 
                              cv2.inRange(hsv, lower_red2, upper_red2))
                              
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_red:
        cnt = max(contours_red, key=cv2.contourArea)
        # Check if it's a circle or rect. Spec says construction is a zone.
        # Let's approximate as a circle for the "safe" zone center
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        xm = x * scale_x
        ym = 5.0 - (y * scale_y)
        rm = radius * max(scale_x, scale_y)
        
        config['zones']['construction'] = {
            'type': 'safe',
            'circle': {'x': float(xm), 'y': float(ym), 'r': float(rm)}
        }
        
        # Add goal marker at center of construction zone
        config['markers'].append({'id': 0, 'x': float(xm), 'y': float(ym), 'type': 'goal'})

    # --- Excavation Zone (Danger) ---
    # Not easily color coded in standard schematic? 
    # Usually it's just a region. Let's assume if we don't find it, we use default?
    # Or maybe it's marked with another color.
    # For now, let's just keep the default excavation zone if not found, 
    # OR we can try to detect text or lines. Too complex for now.
    # Let's add a default excavation zone if not detected.
    config['zones']['excavation'] = {
        'type': 'danger',
        'polygon': [[5.0, 0.0], [9.0, 0.0], [9.0, 2.5], [5.5, 2.0]]
    }

    return config
