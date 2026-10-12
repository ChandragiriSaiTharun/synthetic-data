"""
YOLOv8 Vehicle Detection Evaluation Script for Synthetic Toll-Plaza Traffic Videos.

Evaluates pretrained YOLOv8 (e.g. yolov8n.pt) on generated synthetic traffic video:
- Filters detections to target vehicle classes: car, motorcycle, bus, truck
- Annotates frames with bounding boxes, labels, and confidence scores
- Exports annotated video to synthetic_toll_dataset/yolo_results/scenario_04_yolo_test.mp4
- Evaluates detection counts across thresholds (0.25, 0.40, 0.50)
- Computes class breakdown and average confidence metrics
"""

import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# COCO Vehicle Classes Mapping
VEHICLE_CLASS_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CLASS_COLORS = {
    "car": (0, 220, 0),          # Green
    "motorcycle": (255, 180, 0),  # Cyan / Light Blue
    "bus": (0, 140, 255),        # Orange
    "truck": (255, 60, 60),      # Blue
}


def run_yolo_detection(
    video_path: str = "synthetic_toll_dataset/videos/scenario_04_lane2_congestion/traffic.mp4",
    model_path: str = "yolov8n.pt",
    output_path: str = "synthetic_toll_dataset/yolo_results/scenario_04_yolo_test.mp4",
    conf_threshold: float = 0.25,
    save_annotated: bool = True,
    device: str = "cpu"
) -> dict:
    """Processes video with YOLOv8, annotates frames, and collects detection statistics."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Load YOLO model
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_writer = None
    if save_annotated:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing '{video_path}' ({total_frames} frames, {width}x{height} @ {fps:.1f} fps) at conf={conf_threshold}...")

    total_detections = 0
    class_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
    all_confs = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model(frame, conf=conf_threshold, verbose=False, device=device)[0]

        frame_annotated = frame.copy() if save_annotated else None

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id in VEHICLE_CLASS_IDS:
                cname = VEHICLE_CLASS_IDS[cls_id]
                total_detections += 1
                class_counts[cname] += 1
                all_confs.append(conf)

                if save_annotated:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    color = CLASS_COLORS.get(cname, (0, 255, 0))

                    # Draw bounding box
                    cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 2)

                    # Label with confidence
                    label = f"{cname} {conf:.2f}"
                    font_scale = 0.5
                    thickness = 1
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    cv2.rectangle(frame_annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(0, y1)), color, -1)
                    cv2.putText(
                        frame_annotated,
                        label,
                        (x1 + 2, max(th, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        thickness,
                        cv2.LINE_AA
                    )

        if save_annotated and out_writer is not None:
            # Overlay summary HUD
            hud_text = f"Frame {frame_idx:04d}/{total_frames} | Conf: {conf_threshold:.2f} | Dets: {total_detections}"
            cv2.putText(frame_annotated, hud_text, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            out_writer.write(frame_annotated)

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames ({total_detections} detections so far)...")

    cap.release()
    if out_writer is not None:
        out_writer.release()
        print(f"Saved annotated video to: {output_path}")

    avg_conf = float(np.mean(all_confs)) if all_confs else 0.0

    stats = {
        "conf_threshold": conf_threshold,
        "total_frames": frame_idx,
        "total_detections": total_detections,
        "class_counts": class_counts,
        "average_confidence": avg_conf,
    }

    return stats


def test_confidence_thresholds(
    video_path: str = "synthetic_toll_dataset/videos/scenario_04_lane2_congestion/traffic.mp4",
    model_path: str = "yolov8n.pt",
    thresholds: list = (0.25, 0.40, 0.50),
    device: str = "cpu"
):
    """Evaluates detection counts across confidence thresholds 0.25, 0.40, 0.50."""
    print("\n" + "=" * 55)
    print("DETECTION THRESHOLD TEST (YOLOv8 Pretrained Baseline)")
    print("=" * 55)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"Loaded {len(frames)} frames for threshold evaluation.\n")

    results_by_thresh = {}
    for thresh in thresholds:
        count = 0
        counts_by_cls = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        all_confs = []
        for frame in frames:
            res = model(frame, conf=thresh, verbose=False, device=device)[0]
            for b in res.boxes:
                cls_id = int(b.cls[0])
                if cls_id in VEHICLE_CLASS_IDS:
                    cname = VEHICLE_CLASS_IDS[cls_id]
                    count += 1
                    counts_by_cls[cname] += 1
                    all_confs.append(float(b.conf[0]))

        avg_c = float(np.mean(all_confs)) if all_confs else 0.0
        results_by_thresh[thresh] = {
            "total": count,
            "counts_by_class": counts_by_cls,
            "avg_conf": avg_c
        }
        print(f"Confidence {thresh:.2f} -> {count:4d} detections | "
              f"Cars: {counts_by_cls['car']:3d}, Trucks: {counts_by_cls['truck']:3d}, "
              f"Buses: {counts_by_cls['bus']:3d}, Motorcycles: {counts_by_cls['motorcycle']:3d} "
              f"(Avg Conf: {avg_c:.3f})")

    print("=" * 55 + "\n")
    return results_by_thresh


def main():
    parser = argparse.ArgumentParser(description="Test pretrained YOLOv8 on synthetic toll-plaza traffic video")
    parser.add_argument("--video", type=str, default="synthetic_toll_dataset/videos/scenario_04_lane2_congestion/traffic.mp4")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--output", type=str, default="synthetic_toll_dataset/yolo_results/scenario_04_yolo_test.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--threshold-test", action="store_true", default=True, help="Run detection threshold test for 0.25, 0.40, 0.50")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # 1. Main video detection & video annotation
    stats = run_yolo_detection(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        conf_threshold=args.conf,
        save_annotated=True,
        device=args.device
    )

    print("\n" + "=" * 45)
    print("YOLO DETECTION SUMMARY (Baseline conf = 0.25)")
    print("=" * 45)
    print(f"Total detections:       {stats['total_detections']}")
    print(f"Car detections:         {stats['class_counts']['car']}")
    print(f"Motorcycle detections:  {stats['class_counts']['motorcycle']}")
    print(f"Bus detections:         {stats['class_counts']['bus']}")
    print(f"Truck detections:       {stats['class_counts']['truck']}")
    print(f"Average confidence:     {stats['average_confidence']:.3f}")
    print("=" * 45)

    # 2. Threshold test
    if args.threshold_test:
        test_confidence_thresholds(
            video_path=args.video,
            model_path=args.model,
            thresholds=[0.25, 0.40, 0.50],
            device=args.device
        )


if __name__ == "__main__":
    main()
