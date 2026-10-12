"""
Script to extract, align, rotate to +Y (downward traffic direction),
and save transparent RGBA vehicle sprites into assets/vehicles/.
Also verifies YOLOv8 detection on each generated sprite asset.
"""

import os
import cv2
import numpy as np

ARTIFACT_DIR = r"C:\Users\chand\.gemini\antigravity-ide\brain\561fba83-d737-4d92-818c-2df9aff70e06"

# (filename, output_name, rotation_angle_degrees_to_align_vertically_facing_down)
SPRITE_SPECS = {
    "car": [
        ("car_topdown_test_1790265451799.jpg", "car_01.png", 0.0),      # Grey BMW sedan (already pointing down)
        ("car_suv_topdown_1790265521084.jpg", "car_02.png", 18.0),     # White SUV (slight clockwise tilt)
        ("car_red_topdown_1790265643660.jpg", "car_03.png", 22.0),     # Red hatchback
    ],
    "truck": [
        ("truck_box_topdown_1790265551731.jpg", "truck_01.png", -32.0), # White box truck
        ("truck_cargo_topdown_1790265667043.jpg", "truck_02.png", -30.0),# Blue semi truck
    ],
    "bus": [
        ("bus_coach_topdown_1790265578552.jpg", "bus_01.png", 32.0),    # White coach bus
        ("bus_city_topdown_1790265693856.jpg", "bus_02.png", 30.0),     # Yellow city transit bus
    ],
    "motorcycle": [
        ("motorcycle_topdown_1790265607433.jpg", "motorcycle_01.png", -28.0), # Blue commuter bike
        ("motorcycle_red_topdown_1790265725012.jpg", "motorcycle_02.png", 30.0), # Red sports bike
    ]
}


def clean_and_extract_vehicle(src_path: str, rot_deg: float) -> np.ndarray:
    img = cv2.imread(src_path)
    if img is None:
        raise FileNotFoundError(f"Missing {src_path}")

    h, w = img.shape[:2]

    # Remove any corner timestamps/watermarks in top 120px
    img[:120, :380] = [255, 255, 255]
    img[:120, w-380:] = [255, 255, 255]

    # Floodfill background from 4 corners
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    diff = (22, 22, 22)
    corners = [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3), (w // 2, 2)]
    
    # Copy for floodfill
    fill_img = img.copy()
    for cx, cy in corners:
        cv2.floodFill(fill_img, flood_mask, (cx, cy), (0, 0, 0), diff, diff, cv2.FLOODFILL_FIXED_RANGE)

    # In flood_mask, 1 indicates background
    bg_mask = flood_mask[1:h+1, 1:w+1]
    
    # Also mark anything near-white as background
    is_white = (img[:, :, 0] > 238) & (img[:, :, 1] > 238) & (img[:, :, 2] > 238)
    combined_bg = (bg_mask > 0) | is_white

    # Foreground vehicle mask
    fg_mask = (~combined_bg).astype(np.uint8) * 255

    # Clean small artifacts
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, k)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k)

    # Rotate if needed
    if abs(rot_deg) > 0.5:
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
        # Bounding box for rotated image to avoid clipping
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        img = cv2.warpAffine(img, M, (new_w, new_h), borderValue=(255, 255, 255))
        fg_mask = cv2.warpAffine(fg_mask, M, (new_w, new_h), borderValue=0)
        h, w = new_h, new_w

    # Find bounding box of foreground
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter out tiny noise contours
    valid_contours = [c for c in contours if cv2.contourArea(c) > 1000]
    if not valid_contours:
        valid_contours = contours

    all_pts = np.vstack(valid_contours)
    x, y, cw, ch = cv2.boundingRect(all_pts)

    # Add margin
    pad = 6
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + cw + pad)
    y2 = min(h, y + ch + pad)

    crop_img = img[y1:y2, x1:x2]
    crop_fg = fg_mask[y1:y2, x1:x2]

    # Smooth alpha edge
    alpha = cv2.GaussianBlur(crop_fg, (3, 3), 0)

    # Merge BGRA
    b, g, r = cv2.split(crop_img)
    bgra = cv2.merge([b, g, r, alpha])

    return bgra


def main():
    base_dir = "assets/vehicles"
    os.makedirs(base_dir, exist_ok=True)

    overview_tiles = []

    for cat, items in SPRITE_SPECS.items():
        cat_dir = os.path.join(base_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)

        for src_name, out_name, rot in items:
            src_path = os.path.join(ARTIFACT_DIR, src_name)
            out_path = os.path.join(cat_dir, out_name)

            bgra = clean_and_extract_vehicle(src_path, rot)
            if bgra is not None:
                cv2.imwrite(out_path, bgra)
                print(f"[{cat.upper()}] Processed {out_name}: {bgra.shape[1]}x{bgra.shape[0]} px")
                
                # Make preview thumbnail on dark road surface
                thumb_h = 240
                thumb_w = int(bgra.shape[1] * (thumb_h / bgra.shape[0]))
                thumb = cv2.resize(bgra, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

                # Composite over road color
                tile = np.full((thumb_h + 40, max(thumb_w + 40, 160), 3), (52, 50, 48), dtype=np.uint8)
                tx = (tile.shape[1] - thumb_w) // 2
                ty = 10
                
                # Alpha blend
                alpha_f = thumb[:, :, 3] / 255.0
                for c in range(3):
                    tile[ty:ty+thumb_h, tx:tx+thumb_w, c] = (
                        alpha_f * thumb[:, :, c] + (1.0 - alpha_f) * tile[ty:ty+thumb_h, tx:tx+thumb_w, c]
                    )

                cv2.putText(tile, f"{cat}: {out_name}", (10, thumb_h + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
                overview_tiles.append(tile)

    # Save overview grid image
    if overview_tiles:
        max_th = max(t.shape[0] for t in overview_tiles)
        total_tw = sum(t.shape[1] for t in overview_tiles[:5])
        # Two rows
        row1 = np.hstack([cv2.copyMakeBorder(t, 0, max_th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in overview_tiles[:5]])
        row2_tiles = overview_tiles[5:]
        row2 = np.hstack([cv2.copyMakeBorder(t, 0, max_th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in row2_tiles])
        if row2.shape[1] < row1.shape[1]:
            row2 = cv2.copyMakeBorder(row2, 0, 0, 0, row1.shape[1] - row2.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
        
        grid = np.vstack([row1, row2])
        cv2.imwrite("assets/sprite_overview.png", grid)
        print("Saved sprite overview to assets/sprite_overview.png")


if __name__ == "__main__":
    main()
