"""
Script to generate a high-clarity visualization of the 3 Lane Polygons
and the Entry, Toll, and Exit Regions of Interest (ROIs).
Outputs an annotated PNG image for research reference and YOLO pipeline alignment.
"""

import os
import cv2
import numpy as np
from src.config import (
    FRAME_WIDTH, FRAME_HEIGHT, LANE_POLYGONS, TOLL_BOOTHS,
    TOLL_STOP_Y, get_lane_rois, export_geometry_json
)
from src.renderer import TollPlazaRenderer


def generate_roi_visualization(output_path: str = "synthetic_toll_dataset/metadata/lane_rois_visualization.png") -> str:
    """Renders the toll plaza scene annotated with all lane and ROI polygons."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    renderer = TollPlazaRenderer()

    # Get static background
    img = renderer.static_background.copy()
    overlay = img.copy()

    # Color palette (BGR)
    lane_colors = {
        1: (240, 180, 50),   # Cyan/Sky
        2: (50, 210, 100),   # Bright Green
        3: (180, 80, 240),   # Purple/Pink
    }

    roi_colors = {
        "entry_roi": (255, 140, 0),   # Blue / Cyan
        "toll_roi":  (30, 30, 240),   # Bright Red
        "exit_roi":  (30, 220, 60),   # Green
    }

    # 1. Fill Lane Polygons with subtle tint
    for lane_id, poly in LANE_POLYGONS.items():
        pts = np.array(poly, dtype=np.int32)
        color = lane_colors[lane_id]
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 3, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

    # 2. Highlight ROIs (Entry, Toll, Exit)
    overlay_roi = img.copy()
    lane_rois = get_lane_rois()

    for lane_id, rois in lane_rois.items():
        for roi_type, poly in rois.items():
            pts = np.array(poly, dtype=np.int32)
            c = roi_colors[roi_type]
            cv2.fillPoly(overlay_roi, [pts], c)
            cv2.polylines(img, [pts], True, c, 2, cv2.LINE_AA)

    cv2.addWeighted(overlay_roi, 0.30, img, 0.70, 0, img)

    # 3. Draw Toll Stop Line in high contrast
    cv2.line(img, (220, int(TOLL_STOP_Y)), (1060, int(TOLL_STOP_Y)), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"TOLL STOP LINE (Y = {int(TOLL_STOP_Y)})", (100, int(TOLL_STOP_Y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    # 4. Annotate Lane Polygons and ROIs
    for lane_id, rois in lane_rois.items():
        # Label Entry ROI
        e_poly = rois["entry_roi"]
        ecx = (e_poly[0][0] + e_poly[1][0]) // 2
        cv2.putText(img, f"L{lane_id} ENTRY", (ecx - 38, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Label Toll ROI
        t_poly = rois["toll_roi"]
        tcx = (t_poly[0][0] + t_poly[1][0]) // 2
        cv2.putText(img, f"L{lane_id} TOLL", (tcx - 32, 445),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Label Exit ROI
        x_poly = rois["exit_roi"]
        xcx = (x_poly[0][0] + x_poly[1][0]) // 2
        cv2.putText(img, f"L{lane_id} EXIT", (xcx - 32, 650),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Legend Box
    lx, ly = 30, 70
    cv2.rectangle(img, (lx, ly), (lx + 270, ly + 210), (20, 24, 28), -1)
    cv2.rectangle(img, (lx, ly), (lx + 270, ly + 210), (120, 130, 140), 2)
    cv2.putText(img, "RESEARCH ROI DEFINITIONS", (lx + 15, ly + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    items = [
        ("Lane 1 Boundary", (240, 180, 50)),
        ("Lane 2 Boundary", (50, 210, 100)),
        ("Lane 3 Boundary", (180, 80, 240)),
        ("Entry ROI (y: 30-140)", (255, 140, 0)),
        ("Toll Stop ROI (y: 400-490)", (30, 30, 240)),
        ("Exit ROI (y: 590-710)", (30, 220, 60)),
        ("Stop Bar (y = 450)", (0, 255, 255)),
    ]

    for idx, (label, color) in enumerate(items):
        item_y = ly + 55 + idx * 21
        cv2.rectangle(img, (lx + 15, item_y - 10), (lx + 30, item_y + 4), color, -1)
        cv2.putText(img, label, (lx + 40, item_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    print(f"ROI Visualization saved to: {output_path}")

    # Also export JSON definition
    json_path = os.path.join(os.path.dirname(output_path), "lanes_config.json")
    export_geometry_json(json_path)
    print(f"Lane geometry exported to: {json_path}")
    return output_path


if __name__ == "__main__":
    generate_roi_visualization()
