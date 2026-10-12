"""
Test realistic vehicle sizing and natural soft shadows on toll plaza road,
and test YOLOv8 detection.
"""

import cv2
import numpy as np
from ultralytics import YOLO

from src.renderer import TollPlazaRenderer
from prepare_aligned_sprites import clean_and_extract_vehicle, ARTIFACT_DIR

model = YOLO("yolov8n.pt")
renderer = TollPlazaRenderer()

# Clean sprites
car_bgra = clean_and_extract_vehicle(rf"{ARTIFACT_DIR}\car_suv_topdown_1790265521084.jpg", 18.0)
truck_bgra = clean_and_extract_vehicle(rf"{ARTIFACT_DIR}\truck_box_topdown_1790265551731.jpg", 0.0)
bus_bgra = clean_and_extract_vehicle(rf"{ARTIFACT_DIR}\bus_city_topdown_1790265693856.jpg", 0.0)
moto_bgra = clean_and_extract_vehicle(rf"{ARTIFACT_DIR}\motorcycle_topdown_1790265607433.jpg", 0.0)


def overlay_realistic(canvas, sprite_bgra, cx, cy, target_w, target_h, angle=0.0):
    resized = cv2.resize(sprite_bgra, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)

    if abs(angle) > 0.5:
        M = cv2.getRotationMatrix2D((target_w/2, target_h/2), angle, 1.0)
        resized = cv2.warpAffine(resized, M, (int(target_w), int(target_h)), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    sw = resized.shape[1]
    sh = resized.shape[0]
    x1 = int(cx - sw / 2)
    y1 = int(cy - sh / 2)
    x2 = x1 + sw
    y2 = y1 + sh

    # Subtle soft contact shadow underneath
    # We create a shadow mask from the sprite alpha, dilated and blurred
    alpha_mask = resized[:, :, 3]
    shadow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated_alpha = cv2.dilate(alpha_mask, shadow_kernel, iterations=2)
    shadow_soft = cv2.GaussianBlur(dilated_alpha, (15, 15), 0)

    # Offset shadow slightly (down 4px, right 3px)
    sh_x1 = x1 + 4
    sh_y1 = y1 + 5
    sh_x2 = sh_x1 + sw
    sh_y2 = sh_y1 + sh

    # Blend shadow
    csx1, csy1 = max(0, sh_x1), max(0, sh_y1)
    csx2, csy2 = min(canvas.shape[1], sh_x2), min(canvas.shape[0], sh_y2)
    if csx2 > csx1 and csy2 > csy1:
        s_patch = canvas[csy1:csy2, csx1:csx2]
        sh_alpha = (shadow_soft[csy1-sh_y1:csy2-sh_y1, csx1-sh_x1:csx2-sh_x1] / 255.0) * 0.40
        sh_alpha = sh_alpha[:, :, np.newaxis]
        shadow_blended = ((1.0 - sh_alpha) * s_patch).astype(np.uint8)
        canvas[csy1:csy2, csx1:csx2] = shadow_blended

    # Blend vehicle sprite
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


def test_sizes():
    for scale_factor in [1.0, 1.4, 1.8, 2.2]:
        test_img = renderer.static_background.copy()

        # Place car in Lane 1 at y=350
        # Realistic proportions: base width ~ 70 * scale_factor, length ~ 140 * scale_factor
        w_car = int(55 * scale_factor)
        h_car = int(105 * scale_factor)
        overlay_realistic(test_img, car_bgra, 380, 350, w_car, h_car, angle=-3.0)

        # Place truck in Lane 2 at y=400
        w_truck = int(70 * scale_factor)
        h_truck = int(140 * scale_factor)
        overlay_realistic(test_img, truck_bgra, 630, 380, w_truck, h_truck, angle=0.0)

        # Place bus in Lane 3 at y=320
        w_bus = int(70 * scale_factor)
        h_bus = int(160 * scale_factor)
        overlay_realistic(test_img, bus_bgra, 860, 320, w_bus, h_bus, angle=5.0)

        # Place motorcycle in Lane 2 at y=160
        w_moto = int(32 * scale_factor)
        h_moto = int(65 * scale_factor)
        overlay_realistic(test_img, moto_bgra, 620, 160, w_moto, h_moto, angle=0.0)

        # Run YOLOv8
        res = model(test_img, conf=0.20, verbose=False)[0]
        dets = []
        for b in res.boxes:
            cname = model.names[int(b.cls[0])]
            conf = float(b.conf[0])
            dets.append(f"{cname}:{conf:.2f}")

        print(f"Scale {scale_factor:.1f} (Car: {w_car}x{h_car}, Truck: {w_truck}x{h_truck}) -> Detections ({len(dets)}): {dets}")
        cv2.imwrite(f"test_scale_{scale_factor:.1f}.png", test_img)


if __name__ == "__main__":
    test_sizes()
