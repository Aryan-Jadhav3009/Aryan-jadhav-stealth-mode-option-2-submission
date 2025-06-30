
from ultralytics import YOLO

# Load detector
model = YOLO("model.pt")

# Classes to keep
VALID_NAMES = ["player", "referee", "goalkeeper"]

def detect_players(frame, conf_th=0.5):
    results = model(frame)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        if name in VALID_NAMES and conf >= conf_th:
            detections.append([x1, y1, x2, y2, conf])
    return detections
