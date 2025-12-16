from ultralytics import YOLO
import math

model = YOLO("runs/detect/yolov9_fuel_detection/weights/best.pt")

def get_fuel_level(image_path):
    result = model(image_path, device="cpu")[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return {"error": "No detections"}

    def get_center(name):
        for box in boxes:
            if result.names[int(box.cls)] == name:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                return [(x1 + x2) / 2, (y1 + y2) / 2]
        return None

    points = {name: get_center(name) for name in ["empty", "full", "needle_center", "needle_tip"]}

    if None in points.values():
        missing = [k for k, v in points.items() if v is None]
        return {"error": f"Missing: {missing}"}

    empty, full, center, tip = points.values()

    def angle(c, p):
        dx, dy = p[0] - c[0], p[1] - c[1]
        return math.degrees(math.atan2(dy, dx)) % 360

    aE, aF, aT = angle(center, empty), angle(center, full), angle(center, tip)

    cw_dist = (aF - aE) % 360
    ccw_dist = (aE - aF) % 360

    if cw_dist <= ccw_dist:
        start, arc = aE, cw_dist
    else:
        start, arc = aE, ccw_dist

    needle_rel = aT - start
    if needle_rel < -180:
        needle_rel += 360
    elif needle_rel > 180:
        needle_rel -= 360

    fuel_pct = (needle_rel / arc * 100) if arc != 0 else 50
    fuel_pct = max(0, min(100, round(fuel_pct, 1)))

    return {"fuel_percentage": fuel_pct}
