"""
Configuration module for the Smart Toll Plaza Traffic Simulator.
Defines scene dimensions, lane geometries, ROIs, vehicle specifications,
and scenario configurations.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

# Video & Display Settings
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
DEFAULT_FPS: int = 30
DEFAULT_DURATION: int = 30  # seconds

# Road and Plaza Layout Coordinates
# Elevated CCTV perspective: road flows from top (entry) to bottom (exit)
# Y-axis milestones:
ENTRY_Y_SPAWN: float = -60.0   # Spawn above visible frame for smooth entry
ENTRY_Y_VISIBLE: float = 10.0   # Visible entry line in camera frame
TOLL_STOP_Y: float = 450.0      # Front bumper stop line at toll booth
TOLL_PASS_Y: float = 490.0      # Vehicle clears toll booth
EXIT_Y_VISIBLE: float = 720.0   # Off the bottom of camera view

# 4-point convex polygon boundaries for the 3 lanes: [TL, TR, BR, BL]
# Road slightly flares towards the bottom to match elevated CCTV perspective
LANE_POLYGONS: Dict[int, List[List[int]]] = {
    1: [[310, 0], [480, 0], [460, 720], [240, 720]],
    2: [[510, 0], [680, 0], [720, 720], [530, 720]],
    3: [[710, 0], [880, 0], [1010, 720], [790, 720]],
}

# Concrete Island Boundaries (dividers where toll booths sit)
ISLAND_POLYGONS: Dict[str, List[List[int]]] = {
    "shoulder_left": [[260, 0], [305, 0], [235, 720], [190, 720]],
    "island_1_2":    [[485, 0], [505, 0], [525, 720], [465, 720]],
    "island_2_3":    [[685, 0], [705, 0], [785, 720], [725, 720]],
    "shoulder_right":[[885, 0], [930, 0], [1060, 720], [1015, 720]],
}

# Toll Booth Structure Positions (center x, y, width, height)
# Booth cabins sit on the divider islands adjacent to the stop line
TOLL_BOOTHS: Dict[int, Dict[str, Any]] = {
    1: {
        "lane_id": 1,
        "name": "Lane 1 (Left)",
        "cabin_center": (495, 435),
        "barrier_pivot": (475, 455),
        "barrier_length": 150,
        "barrier_arm_end": (330, 455),
        "label_pos": (370, 395),
    },
    2: {
        "lane_id": 2,
        "name": "Lane 2 (Center)",
        "cabin_center": (745, 435),
        "barrier_pivot": (725, 455),
        "barrier_length": 150,
        "barrier_arm_end": (580, 455),
        "label_pos": (625, 395),
    },
    3: {
        "lane_id": 3,
        "name": "Lane 3 (Right)",
        "cabin_center": (765, 435),  # or right shoulder side
        "barrier_pivot": (785, 455),
        "barrier_length": 150,
        "barrier_arm_end": (930, 455),
        "label_pos": (885, 395),
    },
}

# Regions of Interest (ROIs) - Defined for the research pipeline
# 1. Global ROIs (horizontal slices across all lanes)
GLOBAL_ROIS: Dict[str, Dict[str, Any]] = {
    "entry_region": {
        "description": "Vehicle entry and detection zone",
        "y_min": 30,
        "y_max": 140,
        "polygon": [[300, 30], [900, 30], [940, 140], [280, 140]],
    },
    "toll_stop_region": {
        "description": "Toll processing and stopping zone",
        "y_min": 400,
        "y_max": 490,
        "polygon": [[260, 400], [980, 400], [250, 490], [1000, 490]],
    },
    "exit_region": {
        "description": "Vehicle exit verification zone",
        "y_min": 590,
        "y_max": 710,
        "polygon": [[240, 590], [1030, 590], [240, 710], [1040, 710]],
    },
}

def get_lane_x_bounds(lane_id: int, y: float) -> Tuple[float, float]:
    """Calculate the left and right x-coordinates of a lane at any y position."""
    poly = LANE_POLYGONS[lane_id]
    # poly: [TL, TR, BR, BL]
    tl, tr, br, bl = poly
    alpha = max(0.0, min(1.0, y / FRAME_HEIGHT))
    x_left = tl[0] + (bl[0] - tl[0]) * alpha
    x_right = tr[0] + (br[0] - tr[0]) * alpha
    return x_left, x_right

def get_lane_center_x(lane_id: int, y: float) -> float:
    """Return the center X coordinate of a lane at a specific Y."""
    x_l, x_r = get_lane_x_bounds(lane_id, y)
    return (x_l + x_r) / 2.0

def get_lane_width(lane_id: int, y: float) -> float:
    """Return width of the lane at y."""
    x_l, x_r = get_lane_x_bounds(lane_id, y)
    return x_r - x_l

def get_lane_rois() -> Dict[int, Dict[str, List[List[int]]]]:
    """Generate per-lane polygon ROIs for entry, toll, and exit zones."""
    rois = {}
    for lane_id in (1, 2, 3):
        # Entry ROI: y from 30 to 140
        e_tl_x, e_tr_x = get_lane_x_bounds(lane_id, 30)
        e_bl_x, e_br_x = get_lane_x_bounds(lane_id, 140)
        
        # Toll ROI: y from 400 to 490
        t_tl_x, t_tr_x = get_lane_x_bounds(lane_id, 400)
        t_bl_x, t_br_x = get_lane_x_bounds(lane_id, 490)
        
        # Exit ROI: y from 590 to 710
        x_tl_x, x_tr_x = get_lane_x_bounds(lane_id, 590)
        x_bl_x, x_br_x = get_lane_x_bounds(lane_id, 710)
        
        rois[lane_id] = {
            "entry_roi": [[int(e_tl_x), 30], [int(e_tr_x), 30], [int(e_br_x), 140], [int(e_bl_x), 140]],
            "toll_roi":  [[int(t_tl_x), 400], [int(t_tr_x), 400], [int(t_br_x), 490], [int(t_bl_x), 490]],
            "exit_roi":  [[int(x_tl_x), 590], [int(x_tr_x), 590], [int(x_br_x), 710], [int(x_bl_x), 710]],
        }
    return rois

# YOLOv8 COCO Class Mapping for Vehicle Detection
YOLO_CLASS_MAPPING: Dict[str, int] = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}

# Perspective Scaling:
# Far away (y ~ 0) scale is ~0.65x; near toll (y ~ 450) is ~1.0x; near bottom (y ~ 720) is ~1.20x
def get_perspective_scale(y: float) -> float:
    """Returns the visual scaling factor based on perspective depth."""
    clamped_y = max(0.0, min(FRAME_HEIGHT, y))
    return 0.65 + 0.55 * (clamped_y / FRAME_HEIGHT)

def get_lane_angle(lane_id: int, y: float = 360.0) -> float:
    """
    Returns the road perspective trajectory angle in degrees:
    Lane 1 tilts slightly left, Lane 2 runs near-straight, Lane 3 tilts right.
    """
    if lane_id == 1:
        return -3.5
    elif lane_id == 2:
        return 1.2
    elif lane_id == 3:
        return 6.5
    return 0.0

# Vehicle Class Specifications & Configurable Traffic Distribution
# Configurable distribution: cars 70%, motorcycles 15%, trucks 10%, buses 5%
VEHICLE_SPECS: Dict[str, Dict[str, Any]] = {
    "car": {
        "name": "car",
        "yolo_class_id": 2,
        "base_length": 120.0,
        "base_width": 62.0,
        "speed_range": (90.0, 130.0),    # px / s
        "accel": 90.0,
        "decel": 140.0,
        "toll_time_range": (2.2, 3.5),
        "min_following_gap": 22.0,
        "weight_in_traffic": 0.70,       # 70% cars
    },
    "motorcycle": {
        "name": "motorcycle",
        "yolo_class_id": 3,
        "base_length": 65.0,
        "base_width": 32.0,
        "speed_range": (95.0, 140.0),
        "accel": 120.0,
        "decel": 160.0,
        "toll_time_range": (1.4, 2.2),
        "min_following_gap": 16.0,
        "weight_in_traffic": 0.15,       # 15% motorcycles
    },
    "truck": {
        "name": "truck",
        "yolo_class_id": 7,
        "base_length": 180.0,
        "base_width": 78.0,
        "speed_range": (65.0, 95.0),
        "accel": 50.0,
        "decel": 90.0,
        "toll_time_range": (4.0, 6.5),
        "min_following_gap": 30.0,
        "weight_in_traffic": 0.10,       # 10% trucks
    },
    "bus": {
        "name": "bus",
        "yolo_class_id": 5,
        "base_length": 215.0,
        "base_width": 80.0,
        "speed_range": (65.0, 90.0),
        "accel": 45.0,
        "decel": 85.0,
        "toll_time_range": (3.8, 6.0),
        "min_following_gap": 32.0,
        "weight_in_traffic": 0.05,       # 5% buses
    },
}

# Scenario Configurations as specified in research requirements
SCENARIO_CONFIGS: Dict[int, Dict[str, Any]] = {
    1: {
        "id": 1,
        "key": "scenario_01_low",
        "name": "Scenario 1 — Low Traffic",
        "description": "Low vehicle volume, smooth flow with minimal queuing across all 3 lanes.",
        "lane_counts": {1: 5, 2: 4, 3: 6},
        "lane_arrival_rate": {1: 0.18, 2: 0.15, 3: 0.22},  # vehicles per second
        "toll_time_multiplier": {1: 1.0, 2: 1.0, 3: 1.0},
    },
    2: {
        "id": 2,
        "key": "scenario_02_medium",
        "name": "Scenario 2 — Medium Traffic",
        "description": "Moderate traffic volume with periodic light queuing at toll booths.",
        "lane_counts": {1: 12, 2: 18, 3: 10},
        "lane_arrival_rate": {1: 0.42, 2: 0.65, 3: 0.35},
        "toll_time_multiplier": {1: 1.0, 2: 1.0, 3: 1.0},
    },
    3: {
        "id": 3,
        "key": "scenario_03_heavy",
        "name": "Scenario 3 — Heavy Traffic",
        "description": "High traffic density across all lanes with extended queues and sustained backlogs.",
        "lane_counts": {1: 25, 2: 18, 3: 30},
        "lane_arrival_rate": {1: 0.88, 2: 0.65, 3: 1.05},
        "toll_time_multiplier": {1: 1.1, 2: 1.05, 3: 1.1},
    },
    4: {
        "id": 4,
        "key": "scenario_04_lane2_congestion",
        "name": "Scenario 4 — Lane 2 Congestion",
        "description": "Significant congestion and persistent queueing isolated to Lane 2, while Lanes 1 and 3 flow with low-to-moderate density.",
        "lane_counts": {1: 8, 2: 30, 3: 10},
        "lane_arrival_rate": {1: 0.28, 2: 1.05, 3: 0.35},
        "toll_time_multiplier": {1: 0.9, 2: 1.35, 3: 0.95}, # Lane 2 has slightly longer processing to ensure queue buildup
    },
    5: {
        "id": 5,
        "key": "scenario_05_balanced",
        "name": "Scenario 5 — Balanced Traffic",
        "description": "Evenly distributed traffic across all 3 lanes with synchronized throughput.",
        "lane_counts": {1: 15, 2: 14, 3: 16},
        "lane_arrival_rate": {1: 0.52, 2: 0.49, 3: 0.55},
        "toll_time_multiplier": {1: 1.0, 2: 1.0, 3: 1.0},
    },
}

def export_geometry_json(filepath: str) -> None:
    """Export lane polygons and ROIs to a standalone JSON file for YOLO evaluation."""
    data = {
        "image_dimensions": {"width": FRAME_WIDTH, "height": FRAME_HEIGHT},
        "lane_polygons": LANE_POLYGONS,
        "island_polygons": ISLAND_POLYGONS,
        "global_rois": GLOBAL_ROIS,
        "lane_rois": get_lane_rois(),
        "toll_stop_y": TOLL_STOP_Y,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
