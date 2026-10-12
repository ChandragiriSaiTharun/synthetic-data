# Synthetic Toll-Plaza Traffic Video Dataset Generator

A high-fidelity Python-based synthetic traffic video dataset generator designed for evaluating and benchmarking computer vision models (such as **YOLOv8**) on toll-plaza surveillance feeds.

The simulator renders realistic multi-lane toll plaza traffic with elevated CCTV camera perspective, dynamic barrier arms, queuing behaviors, and complete ground-truth annotations for vehicle detection, tracking, queue-length estimation, and toll throughput analysis.

---

## Key Features

1. **Realistic CCTV Vehicle Rendering**:
   - High-fidelity transparent sprite assets across 4 vehicle classes: `car`, `truck`, `bus`, and `motorcycle`.
   - Perspective scaling based on camera distance ($y$-coordinate depth scaling).
   - Traffic direction alignment matching 3-lane flare angles.
   - Natural soft contact shadows underneath vehicles.
   - Natural vehicle-to-vehicle occlusion ordering.

2. **Full Ground-Truth Generation**:
   - Per-frame vehicle bounding boxes `[x1, y1, x2, y2]`.
   - Vehicle tracking IDs, vehicle types, and lane assignments.
   - Entry frame, exit frame, waiting time, and queue position metrics.
   - Formatted CSV exports for detection evaluation against YOLOv8.

3. **5 Traffic Scenarios**:
   - **Scenario 1 — Low Traffic**: Smooth flow, minimal queuing across 3 lanes.
   - **Scenario 2 — Medium Traffic**: Moderate volume, periodic light queuing.
   - **Scenario 3 — Heavy Traffic**: High density, extended queues across all lanes.
   - **Scenario 4 — Lane 2 Congestion**: Asymmetric congestion with sustained backlog isolated to Lane 2.
   - **Scenario 5 — Balanced Traffic**: Synchronized throughput evenly distributed.

4. **YOLOv8 Detection Baseline Evaluation**:
   - Built-in validation script `test_yolo_detection.py` using pretrained `yolov8n.pt`.
   - COCO class mapping: `car` (2), `motorcycle` (3), `bus` (5), `truck` (7).
   - Automated threshold evaluation at confidence 0.25, 0.40, and 0.50.
   - Generates annotated evaluation video with bounding boxes and confidence scores.

---

## Directory Structure

```
├── assets/
│   └── vehicles/
│       ├── car/          # Transparent PNG vehicle sprites (sedans, SUVs, hatchbacks)
│       ├── truck/        # Box delivery trucks, freight haulers
│       ├── bus/          # City transit buses, coach buses
│       └── motorcycle/   # Commuter bikes, sport motorcycles
├── src/
│   ├── asset_manager.py  # Discovers, caches, scales, and composites vehicle sprites
│   ├── booth.py          # Toll booth structure, barrier arms, and signals
│   ├── config.py         # Plaza dimensions, lane geometries, ROIs, vehicle specs
│   ├── ground_truth.py   # Ground-truth metrics tracking and CSV export
│   ├── lane.py           # Queue management, car-following physics
│   ├── renderer.py       # High-fidelity CCTV background, markings, and vehicle rendering
│   ├── simulator.py      # Simulation loop and kinematic state management
│   └── vehicle.py        # Vehicle dataclass, state, and bounding box computation
├── synthetic_toll_dataset/
│   ├── annotations/      # Frame-by-frame vehicle bounding box CSVs
│   ├── ground_truth/     # Per-vehicle aggregate summary CSVs
│   ├── metadata/         # Scene geometry JSON, ROI maps, scenario descriptions
│   └── videos/           # Generated MP4 traffic simulation videos
├── generate_dataset.py   # Dataset generator CLI
├── test_yolo_detection.py# YOLOv8 baseline detection and video annotation script
├── evaluate_ground_truth.py # Precision, Recall, and mAP evaluation
├── visualize_rois.py     # Plaza ROI geometry visualization tool
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ChandragiriSaiTharun/synthetic-data.git
   cd synthetic-data
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Generate Synthetic Traffic Videos

Generate a specific scenario (e.g. Scenario 4, 30 seconds at 30 FPS):
```bash
python generate_dataset.py --scenario 4 --duration 30 --fps 30 --seed 42
```

Generate all 5 scenarios with ROI visualization:
```bash
python generate_dataset.py --scenario all --duration 30 --fps 30 --draw-rois
```

### 2. Run Pretrained YOLOv8 Evaluation

Evaluate the generated video with `yolov8n.pt`:
```bash
python test_yolo_detection.py --video synthetic_toll_dataset/videos/scenario_04_lane2_congestion/traffic.mp4 --conf 0.25
```

This will:
- Process the video frame-by-frame with YOLOv8.
- Filter detections to `car`, `truck`, `bus`, and `motorcycle`.
- Draw bounding boxes and confidence scores.
- Save the annotated video to `synthetic_toll_dataset/yolo_results/scenario_04_yolo_test.mp4`.
- Report detection counts across confidence thresholds (0.25, 0.40, 0.50).

### 3. Evaluate Ground Truth Against Detections

Compare YOLO detections against the simulator's exact ground-truth CSV:
```bash
python evaluate_ground_truth.py --scenario 4
```

---

## Adding Custom Vehicle Assets

Place transparent PNG or WebP images into the corresponding category folder:
```
assets/vehicles/car/
assets/vehicles/truck/
assets/vehicles/bus/
assets/vehicles/motorcycle/
```
The asset manager automatically discovers all images in these folders and dynamically selects appearances during simulation.

---

## License

MIT License. Designed for smart traffic research, automated toll lane analysis, and computer vision benchmarking.
