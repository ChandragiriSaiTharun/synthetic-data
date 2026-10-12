"""
Test script to composite the realistic sprites onto the toll plaza road background
at different distances (far, middle, near) and run YOLOv8 detection.
"""

import cv2
import numpy as np
from ultralytics import YOLO

from src.renderer import TollPlazaRenderer
from prepare_aligned_sprites import clean_and_extract_vehicle, ARTIFACT_DIR

# Load model
model = YOLO("yolov8n.pt")
renderer = TollPlazaRenderer()
bg = renderer.static_background.copy()

# Sprite sources and optimal angles
SPRITES = {
    "car": (r"C:\Users\chand\.gemini\antigravity-ide\brain\561fba83-d737-4d92-818c-2df9aff70e06\car_suv_topdown_1790265521084.jpg", 40.0, 75, 38),
    "truck": (r"C:\Users\chand\.gemini\antigravity-ide\brain\561fba83-d737-4d92-818c-2df9aff70e06\truck_box_topdown_1790265551731.jpg", 0.0, 115, 52),
    "bus": (r"C:\Users\chand\.gemini\antigravity-ide\brain\561fba83-d737-4d92-818c-2df9aff70e06\bus_city_topdown_1790265693856.jpg", 0.0, 130, 50),
    "motorcycle": (r"C:\Users\chand\.gemini\antigravity-ide\brain\561fba83-d737-4d92-818c-2df9aff70e06\motorcycle_topdown_1790265607433.jpg", 0.0, 42, 22),
}

processed_sprites = {}
for name, (path, rot, bl, bw) in SPRITES.items():
    bgra = clean_and_extract_vehicle(path, rot)
    processed_sprites[name] = (bgra, bl, bw)


def overlay_sprite(canvas, sprite_bgra, cx, cy, target_w, target_h, angle=0.0):
    # Resize sprite to target_w, target_h
    resized = cv2.resize(sprite_bgra, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)
    
    if abs(angle) > 0.5:
        M = cv2.getRotationMatrix2D((target_w/2, target_h/2), angle, 1.0)
        resized = cv2.warpAffine(resized, M, (int(target_w), int(target_h)), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    # Add soft drop shadow
    sh_w, sh_h = int(target_w * 0.95), int(target_h * 0.95)
    shadow_mask = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.float32)
    sx1 = int(cx - sh_w/2 + 6)
    sy1 = int(cy - sh_h/2 + 8)
    sx2 = sx1 + sh_w
    sy2 = sy1 + sh_h
    if 0 <= sx1 and sx2 < canvas.shape[1] and 0 <= sy1 and sy2 < canvas.shape[0]:
        cv2.ellipse(canvas, ((sx1+sx2)//2, (sy1+sy2)//2), (sh_w//2, sh_h//2), 0, 0, 360, (20, 20, 20), -1)

    # Blending
    sw = resized.shape[1]
    sh = resized.shape[0]
    x1 = int(cx - sw / 2)
    y1 = int(cy - sh / 2)
    x2 = x1 + sw
    y2 = y1 + sh

    # Clip to canvas
    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(canvas.shape[1], x2)
    cy2 = min(canvas.shape[0], y2)

    ox1 = cx1 - x1
    oy1 = cy1 - y1
    ox2 = ox1 + (cx2 - cx1)
    oy2 = oy1 + (cy2 - cy1)

    if cx2 > cx1 and cy2 > cy1:
        patch = canvas[cy1:cy2, cx1:cx2]
        sprite_patch = resized[oy1:oy2, ox1:ox2]
        alpha = (sprite_patch[:, :, 3] / 255.0)[:, :, np.newaxis]
        bgr = sprite_patch[:, :, :3]
        blended = (alpha * bgr + (1.0 - alpha) * patch).astype(np.uint8)
        canvas[cy1:cy2, cx1:cx2] = blended


# Test composite 1: 4 vehicles in lanes
test_img = bg.copy()
# Lane 1 (x ~ 400 at top, 360 at mid, 330 at bottom)
# Car at mid y=300
sp, bl, bw = processed_sprites["car"]
overlay_sprite(test_img, sp, 380, 260, bw * 1.0, bl * 1.0)

# Truck in Lane 2 at y=420 (toll approach)
sp_t, bl_t, bw_t = processed_sprites["truck"]
overlay_sprite(test_img, sp_t, 630, 350, bw_t * 1.0, bl_t * 1.0)

# Bus in Lane 3 at y=220 (mid-distance)
sp_b, bl_b, bw_b = processed_sprites["bus"]
overlay_sprite(test_img, sp_b, 850, 240, bw_b * 0.9, bl_b * 0.9)

# Motorcycle in Lane 2 at y=140 (far distance)
sp_m, bl_m, bw_m = processed_sprites["motorcycle"]
overlay_sprite(test_img, sp_m, 620, 140, bw_m * 0.8, bl_m * 0.8)

# Car near exit in Lane 1 at y=580
overlay_sprite(test_img, sp, 350, 580, bw * 1.25, bl * 1.25)

cv2.imwrite("test_composite_scene.png", test_img)
print("Saved test_composite_scene.png")

# Run YOLOv8 on this composite scene
results = model(test_img, conf=0.25)[0]
print("\n--- YOLOv8 Detections on Composite Toll Scene ---")
print(f"Total detections: {len(results.boxes)}")
for b in results.boxes:
    cls_id = int(b.cls[0])
    cname = model.names[cls_id]
    conf = float(b.conf[0])
    box = [int(v) for v in b.xyxy[0].tolist()]
    print(f"Detected {cname} (class {cls_id}) with confidence {conf:.3f} at box {box}")
