"""
Dataset Generator Entry Point for Smart Toll Plaza Traffic Simulation.
Supports generating individual scenarios or the entire 5-scenario benchmark suite.

Usage:
    python generate_dataset.py --scenario 4
    python generate_dataset.py --scenario all --duration 30 --fps 30 --seed 42
"""

import os
import sys
import argparse
from typing import List, Dict, Any

from src.config import SCENARIO_CONFIGS, DEFAULT_FPS, DEFAULT_DURATION
from src.simulator import TrafficSimulator
from src.ground_truth import save_scenario_metadata
from visualize_rois import generate_roi_visualization


def run_scenario(
    scenario_id: int,
    output_base: str,
    duration: float,
    fps: int,
    seed: int,
    overlay_rois: bool = False
) -> Dict[str, Any]:
    """Generates the MP4 video, vehicle ground truth, and annotations for one scenario."""
    cfg = SCENARIO_CONFIGS[scenario_id]
    key = cfg["key"]
    name = cfg["name"]

    # Paths matching specified structure
    video_dir = os.path.join(output_base, "videos", key)
    gt_dir = os.path.join(output_base, "ground_truth")
    ann_dir = os.path.join(output_base, "annotations")

    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    video_path = os.path.join(video_dir, "traffic.mp4")
    # For scenario 1: scenario_01.csv, scenario 4: scenario_04.csv
    file_prefix = f"scenario_{scenario_id:02d}.csv"
    gt_path = os.path.join(gt_dir, file_prefix)
    ann_path = os.path.join(ann_dir, file_prefix)

    print("\n" + "="*70)
    print(f" GENERATING: {name.upper()}")
    print(f" Target Vehicle Volume: {cfg['lane_counts']}")
    print(f" Video Path: {video_path}")
    print(f" Random Seed: {seed} | Duration: {duration}s | FPS: {fps}")
    print("="*70)

    sim = TrafficSimulator(
        scenario_id=scenario_id,
        duration=duration,
        fps=fps,
        seed=seed,
        draw_rois=overlay_rois
    )

    summary = sim.run_simulation(
        output_video_path=video_path,
        export_gt_path=gt_path,
        export_annotations_path=ann_path
    )

    metadata_entry = {
        "scenario_id": scenario_id,
        "scenario_key": key,
        "scenario_name": name,
        "duration_sec": duration,
        "fps": fps,
        "random_seed": seed,
        "target_lane_1": cfg["lane_counts"].get(1, 0),
        "target_lane_2": cfg["lane_counts"].get(2, 0),
        "target_lane_3": cfg["lane_counts"].get(3, 0),
        "total_scheduled": summary["total_vehicles_scheduled"],
        "total_spawned": summary["total_vehicles_spawned"],
        "total_completed": summary["total_vehicles_completed"],
        "max_queue_lane_1": summary["max_queue_per_lane"][1],
        "max_queue_lane_2": summary["max_queue_per_lane"][2],
        "max_queue_lane_3": summary["max_queue_per_lane"][3],
        "video_file": os.path.relpath(video_path, output_base),
        "ground_truth_file": os.path.relpath(gt_path, output_base),
        "annotations_file": os.path.relpath(ann_path, output_base),
    }
    return metadata_entry


def main():
    parser = argparse.ArgumentParser(description="Synthetic Toll Plaza Traffic Dataset Generator")
    parser.add_argument(
        "--scenario",
        type=str,
        default="4",
        help="Scenario ID to generate (1-5) or 'all' to generate the complete dataset. Default is '4'."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help=f"Simulation duration in seconds. Default is {DEFAULT_DURATION}s."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Video frames per second. Default is {DEFAULT_FPS}."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reproducibility. Default is 42."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="synthetic_toll_dataset",
        help="Root output directory. Default is 'synthetic_toll_dataset'."
    )
    parser.add_argument(
        "--overlay_rois",
        action="store_true",
        help="Overlay semi-transparent ROI polygons directly onto video frames."
    )

    args = parser.parse_args()

    # Create metadata directory and generate ROI visualization
    meta_dir = os.path.join(args.output_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    roi_img_path = os.path.join(meta_dir, "lane_rois_visualization.png")
    generate_roi_visualization(roi_img_path)

    scenario_arg = args.scenario.strip().lower()
    if scenario_arg == "all":
        scenarios_to_run = [1, 2, 3, 4, 5]
    else:
        try:
            sc_id = int(scenario_arg)
            if sc_id not in SCENARIO_CONFIGS:
                raise ValueError()
            scenarios_to_run = [sc_id]
        except ValueError:
            print(f"Error: Invalid scenario '{args.scenario}'. Choose from 1, 2, 3, 4, 5, or 'all'.")
            sys.exit(1)

    metadata_list = []
    # If scenarios.csv already exists, load existing entries so single runs can merge or update
    meta_csv_path = os.path.join(meta_dir, "scenarios.csv")
    existing_meta = {}
    if os.path.exists(meta_csv_path):
        import pandas as pd
        try:
            old_df = pd.read_csv(meta_csv_path)
            for _, row in old_df.iterrows():
                existing_meta[int(row["scenario_id"])] = row.to_dict()
        except Exception:
            pass

    for sc_id in scenarios_to_run:
        meta_entry = run_scenario(
            scenario_id=sc_id,
            output_base=args.output_dir,
            duration=args.duration,
            fps=args.fps,
            seed=args.seed,
            overlay_rois=args.overlay_rois
        )
        existing_meta[sc_id] = meta_entry

    # Save consolidated metadata
    final_meta_list = [existing_meta[k] for k in sorted(existing_meta.keys())]
    save_scenario_metadata(final_meta_list, meta_csv_path)
    print(f"\nConsolidated metadata saved to: {meta_csv_path}")
    print("\nDataset generation completed successfully!")


if __name__ == "__main__":
    main()
