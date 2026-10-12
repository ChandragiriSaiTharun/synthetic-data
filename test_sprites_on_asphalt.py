"""
Test YOLOv8 detection on each of the exported transparent PNG vehicle sprites.
"""

import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

sprites = sorted(glob.glob("assets/vehicles/*/*.png"))
print(f"Testing {len(sprites)} sprite assets with YOLOv8...\n")

for sprite_path in sprites:
    sp = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
    if sp is None or sp.shape[2] != 4:
        print(f"Invalid sprite: {sprite_path}")
        continue

    cat = os.path.basename(os.path.dirname(sprite_path))
    name = os.path.basename(sprite_path)

    # Composite on a 640x640 realistic asphalt road background
    canvas = np.full((640, 640, 3), (52, 50, 48), dtype=np.uint8)
    
    # Scale sprite to ~ 260-320 px height
    scale = 280.0 / max(sp.shape[0], sp.shape[1])
    target_w = int(sp.shape[1] * scale)
    target_h = int(sp.shape[0] * scale)
    resized = cv2.resize(sp, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Center in canvas
    x1 = (640 - target_w) // 2
    y1 = (640 - target_h) // 2
    x2 = x1 + target_w
    y2 = y1 + target_h

    alpha = (resized[:, :, 3] / 255.0)[:, :, np.newaxis]
    bgr = resized[:, :, :3]
    canvas[y1:y2, x1:x2] = (alpha * bgr + (1.0 - alpha) * canvas[y1:y2, x1:x2]).astype(np.uint8)

    res = model(canvas, conf=0.20, verbose=False)[0]
    preds = []
    for b in res.boxes:
        cname = model.names[int(b.cls[0])]
        conf = float(b.conf[0])
        preds.append(f"{cname} ({conf:.2f})")

    status = "DETECTED: " + ", ".join(preds) if preds else "NO DETECTION"
    print(f"[{cat.upper():10s}] {name:15s} -> {status}")
