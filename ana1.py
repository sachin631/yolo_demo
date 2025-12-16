from ultralytics import YOLO
import math

model = YOLO("runs/detect/yolov9_fuel_detection/weights/best.pt")

def get_fuel_level(image_path):
    """Universal fuel calculation with empty as start reference"""
    
    result = model(image_path, device="cpu")[0]
    boxes = result.boxes
    
    if boxes is None or len(boxes) == 0:
        return "No detections"
    
    # Get center points
    def get_center(name):
        for box in boxes:
            if result.names[int(box.cls)] == name:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                return [(x1 + x2) / 2, (y1 + y2) / 2]
        return None
    
    # Get all points
    points = {name: get_center(name) for name in ["empty", "full", "needle_center", "needle_tip"]}
    
    # Check if all found
    if None in points.values():
        missing = [k for k, v in points.items() if v is None]
        return f"Missing: {missing}"
    
    empty, full, center, tip = points.values()
    
    # Angle calculation
    def angle(c, p):
        """Angle from center c to point p (0-360°)"""
        dx, dy = p[0] - c[0], p[1] - c[1]
        return math.degrees(math.atan2(dy, dx)) % 360
    
    aE, aF, aT = angle(center, empty), angle(center, full), angle(center, tip)
    
    # 🔥 UNIVERSAL CALCULATION (Empty always as start reference)
    # Find the SHORTEST path from empty to full (either direction)
    cw_dist = (aF - aE) % 360  # Clockwise distance from empty to full
    ccw_dist = (aE - aF) % 360  # Counterclockwise distance from empty to full
    
    # Choose the direction that gives shortest distance
    if cw_dist <= ccw_dist:
        # Clockwise: Empty → Full is clockwise
        start, end = aE, aF
        arc = cw_dist
    else:
        # Counterclockwise: Empty → Full is counterclockwise
        start, end = aE, aF - 360  # Go negative direction
        arc = ccw_dist
    
    # Needle position relative to empty
    needle_rel = aT - start
    if needle_rel < -180:
        needle_rel += 360
    elif needle_rel > 180:
        needle_rel -= 360
    
    # Calculate percentage (0% at empty, 100% at full)
    fuel_pct = (needle_rel / arc * 100) if arc != 0 else 50
    
    # Clamp to 0-100
    fuel_pct = max(0, min(100, round(fuel_pct, 1)))
    
    return fuel_pct

# Test
image_path = "test/images/images-11-_jpeg.rf.45f7d7f3de1e21dc8331a83e71aa6ae4.jpg"
percentage = get_fuel_level(image_path)

if isinstance(percentage, (int, float)):
    print(f"\n⛽ Fuel: {percentage}%")
    bars = int(percentage / 5)
    print(f"[{'█' * bars}{'░' * (20 - bars)}]")
else:
    print(f"Error: {percentage}")