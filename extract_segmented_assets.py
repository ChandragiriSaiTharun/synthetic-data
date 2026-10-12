"""
Script using YOLOv8-seg to automatically extract clean, pixel-perfect alpha-masked
vehicle sprites from all generated high-fidelity vehicle source images.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

seg_model = YOLO("yolov8n-seg.pt")
ARTIFACT_DIR = r"C:\Users\chand\.gemini\antigravity-ide\brain\561fba83-d737-4d92-818c-2df9aff70e06"

# Sources: (artifact_filename, category, output_filename, target_class_id, rotation_deg)
# COCO class IDs: 2: car, 7: truck, 5: bus, 3: motorcycle
ASSET_SOURCES = [
    # CARS
    ("car_white_cctv_1790266338421.jpg", "car", "car_01.png", 2, 35.0),
    ("car_cctv_perspective_1790266241170.jpg", "car", "car_02.png", 2, 33.0),
    ("car_red_cctv_1790266424556.jpg", "car", "car_03.png", 2, 32.0),
    ("car_topdown_test_1790265451799.jpg", "car", "car_04.png", None, 0.0),
    ("car_suv_topdown_1790265521084.jpg", "car", "car_05.png", None, 18.0),

    # TRUCKS
    ("truck_box_topdown_1790265551731.jpg", "truck", "truck_01.png", 7, 0.0),
    ("truck_cargo_topdown_1790265667043.jpg", "truck", "truck_02.png", 7, 0.0),

    # BUSES
    ("bus_city_topdown_1790265693856.jpg", "bus", "bus_01.png", 5, 0.0),
    ("bus_coach_topdown_1790265578552.jpg", "bus", "bus_02.png", 5, 0.0),

    # MOTORCYCLES
    ("motorcycle_topdown_1790265607433.jpg", "motorcycle", "motorcycle_01.png", 3, 0.0),
    ("motorcycle_red_topdown_1790265725012.jpg", "motorcycle", "motorcycle_02.png", 3, 0.0),
]


def extract_vehicle_sprite(src_path: str, target_class: int, rot_deg: float) -> np.ndarray:
    img = cv2.imread(src_path)
    if img is None:
        print(f"Error loading {src_path}")
        return None

    h, w = img.shape[:2]
    # Mask corner text
    img[:120, :380] = [255, 255, 255]
    img[:120, w-380:] = [255, 255, 255]

    # Run segmentation model
    res = seg_model(img, conf=0.20, verbose=False)[0]

    mask = None
    if res.masks is not None and len(res.masks) > 0:
        # Find mask matching target class or highest confidence vehicle
        best_idx = 0
        if target_class is not None:
            for idx, box in enumerate(res.boxes):
                if int(box.cls[0]) == target_class:
                    best_idx = idx
                    break
        raw_mask = res.masks.data[best_idx].cpu().numpy()
        mask = cv2.resize((raw_mask * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        # Dilate slightly to preserve soft boundaries / mirrors / lights
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, k, iterations=1)
    else:
        # Fallback to color/luminance thresholding if no YOLO mask
        is_bg = (img[:, :, 0] > 235) & (img[:, :, 1] > 235) & (img[:, :, 2] > 235)
        mask = (~is_bg).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    # Rotate if needed to align vehicle
    if abs(rot_deg) > 0.5:
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        img = cv2.warpAffine(img, M, (new_w, new_h), borderValue=(255, 255, 255))
        mask = cv2.warpAffine(mask, M, (new_w, new_h), borderValue=0)
        h, w = new_h, new_w

    # Find tight bounding box of vehicle mask
    ys, xs = np.where(mask > 40)
    if len(ys) == 0:
        return None

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    pad = 6
    y1 = max(0, y1 - pad)
    y2 = min(h, y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(w, x2 + pad)

    crop_img = img[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    # Gaussian blur the alpha mask edge for natural anti-aliasing
    alpha = cv2.GaussianBlur(crop_mask, (3, 3), 0)

    # BGRA output
    b, g, r = cv2.split(crop_img)
    bgra = cv2.merge([b, g, r, alpha])

    return bgra


def main():
    base_dir = "assets/vehicles"
    for src_name, cat, out_name, tgt_cls, rot in ASSET_SOURCES:
        cat_dir = os.path.join(base_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)
        src_path = os.path.join(ARTIFACT_DIR, src_name)
        out_path = os.path.join(cat_dir, out_name)

        bgra = extract_vehicle_sprite(src_path, tgt_cls, rot)
        if bgra is not None:
            cv2.imwrite(out_path, bgra)
            print(f"[{cat.upper()}] Exported {out_name}: {bgra.shape[1]}x{bgra.shape[0]} px")
        else:
            print(f"Failed to process {src_name}")


if __name__ == "__main__":
    main()
