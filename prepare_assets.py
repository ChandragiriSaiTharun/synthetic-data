"""
Script to extract, crop, align, and generate transparent RGBA PNG vehicle sprites
from the generated realistic overhead vehicle images.
"""

import os
import cv2
import numpy as np

ARTIFACT_DIR = r"C:\Users\chand\.gemini\antigravity-ide\brain\561fba83-d737-4d92-818c-2df9aff70e06"

SOURCES = {
    "car": [
        ("car_topdown_test_1790265451799.jpg", "car_01.png", 0),
        ("car_suv_topdown_1790265521084.jpg", "car_02.png", 0),
        ("car_red_topdown_1790265643660.jpg", "car_03.png", 0),
    ],
    "truck": [
        ("truck_box_topdown_1790265551731.jpg", "truck_01.png", 0),
        ("truck_cargo_topdown_1790265667043.jpg", "truck_02.png", 0),
    ],
    "bus": [
        ("bus_coach_topdown_1790265578552.jpg", "bus_01.png", 0),
        ("bus_city_topdown_1790265693856.jpg", "bus_02.png", 0),
    ],
    "motorcycle": [
        ("motorcycle_topdown_1790265607433.jpg", "motorcycle_01.png", 0),
        ("motorcycle_red_topdown_1790265725012.jpg", "motorcycle_02.png", 0),
    ]
}


def process_image(src_path: str, out_path: str, extra_rot: float = 0.0) -> None:
    img = cv2.imread(src_path)
    if img is None:
        print(f"Error loading {src_path}")
        return

    h, w = img.shape[:2]

    # Mask out corner timestamps if present in top corners
    img[:120, :350] = [255, 255, 255]
    img[:120, w-350:] = [255, 255, 255]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Segment vehicle: background is near-white (> 242)
    # Contact shadows under car might be grey (180-240). We want the vehicle body and its immediate contact shadow.
    # Threshold where white background is identified
    is_bg = (img[:, :, 0] > 242) & (img[:, :, 1] > 242) & (img[:, :, 2] > 242)
    fg_mask = (~is_bg).astype(np.uint8) * 255

    # Clean mask using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Find the largest contour (the vehicle)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found in {src_path}")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest_contour)

    # Add margin
    pad = 10
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + cw + pad)
    y2 = min(h, y + ch + pad)

    cropped_img = img[y1:y2, x1:x2]
    cropped_mask = fg_mask[y1:y2, x1:x2]

    # Smooth the alpha mask edge with Gaussian blur
    alpha = cv2.GaussianBlur(cropped_mask, (3, 3), 0)

    # Create BGRA 4-channel image
    b, g, r = cv2.split(cropped_img)
    rgba = cv2.merge([b, g, r, alpha])

    # If orientation needs rotation so front is pointing downward (+Y):
    # Check width vs height: if width > height, rotate 90 deg
    if rgba.shape[1] > rgba.shape[0]:
        rgba = cv2.rotate(rgba, cv2.ROTATE_90_CLOCKWISE)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, rgba)
    print(f"Saved: {out_path} ({rgba.shape[1]}x{rgba.shape[0]})")


def main():
    base_assets = "assets/vehicles"
    for category, items in SOURCES.items():
        for filename, outname, rot in items:
            src_file = os.path.join(ARTIFACT_DIR, filename)
            out_file = os.path.join(base_assets, category, outname)
            process_image(src_file, out_file, rot)


if __name__ == "__main__":
    main()
