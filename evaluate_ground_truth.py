"""
Evaluation module for Smart Toll Plaza synthetic ground-truth dataset.
Analyzes ground-truth CSV files and frame annotations to compute per-lane and per-scenario metrics:
- Vehicle counts per lane
- Average and maximum waiting time per lane
- Maximum queue length per lane
- Average vehicle speed
- Lane occupancy / traffic density
- Generates evaluation_summary.csv

Usage:
    python evaluate_ground_truth.py
    python evaluate_ground_truth.py --scenario 4
    python evaluate_ground_truth.py --gt_dir synthetic_toll_dataset/ground_truth
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def evaluate_scenario_ground_truth(
    gt_csv_path: str,
    ann_csv_path: str = None,
    fps: int = 30
) -> pd.DataFrame:
    """
    Computes lane-level and scenario-level metrics from ground-truth CSVs.
    """
    if not os.path.exists(gt_csv_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_csv_path}")

    df_gt = pd.read_csv(gt_csv_path)
    if df_gt.empty:
        print(f"Warning: {gt_csv_path} is empty.")
        return pd.DataFrame()

    scenario_id = df_gt["scenario_id"].iloc[0]

    # Optional speed estimation from frame annotations if available
    speed_per_vehicle = {}
    if ann_csv_path and os.path.exists(ann_csv_path):
        df_ann = pd.read_csv(ann_csv_path)
        if not df_ann.empty:
            for vid, group in df_ann.groupby("vehicle_id"):
                if len(group) >= 2:
                    group_sorted = group.sort_values("frame")
                    dy = group_sorted["y2"].iloc[-1] - group_sorted["y2"].iloc[0]
                    d_frames = group_sorted["frame"].iloc[-1] - group_sorted["frame"].iloc[0]
                    if d_frames > 0:
                        speed_px_s = (dy / d_frames) * fps
                        speed_per_vehicle[vid] = max(0.0, speed_px_s)

    results = []

    for lane in sorted(df_gt["lane"].unique()):
        lane_df = df_gt[df_gt["lane"] == lane]
        veh_count = len(lane_df)
        avg_wait = lane_df["waiting_time"].mean() if veh_count > 0 else 0.0
        max_wait = lane_df["waiting_time"].max() if veh_count > 0 else 0.0
        avg_stop_dur = lane_df["stopping_duration"].mean() if veh_count > 0 else 0.0
        max_queue = lane_df["max_queue_pos"].max() if veh_count > 0 else 0

        # Calculate average speed for lane
        lane_speeds = [speed_per_vehicle[v] for v in lane_df["vehicle_id"] if v in speed_per_vehicle]
        avg_speed = np.mean(lane_speeds) if lane_speeds else 0.0

        # Traffic density proxy: vehicle throughput density per minute of video
        # or normalized queuing intensity
        traffic_density = round(veh_count / (avg_speed + 1e-3) * 10.0, 3)

        results.append({
            "scenario": int(scenario_id),
            "lane": int(lane),
            "vehicle_count": int(veh_count),
            "avg_waiting_time": round(float(avg_wait), 2),
            "max_waiting_time": round(float(max_wait), 2),
            "avg_toll_duration": round(float(avg_stop_dur), 2),
            "max_queue_size": int(max_queue),
            "avg_speed_px_s": round(float(avg_speed), 2),
            "traffic_density": traffic_density
        })

    # Add overall scenario aggregate row
    total_vehs = len(df_gt)
    overall_avg_wait = df_gt["waiting_time"].mean() if total_vehs > 0 else 0.0
    overall_max_queue = df_gt["max_queue_pos"].max() if total_vehs > 0 else 0
    all_speeds = list(speed_per_vehicle.values())
    overall_avg_speed = np.mean(all_speeds) if all_speeds else 0.0

    results.append({
        "scenario": int(scenario_id),
        "lane": "ALL",
        "vehicle_count": int(total_vehs),
        "avg_waiting_time": round(float(overall_avg_wait), 2),
        "max_waiting_time": round(float(df_gt["waiting_time"].max()), 2),
        "avg_toll_duration": round(float(df_gt["stopping_duration"].mean()), 2),
        "max_queue_size": int(overall_max_queue),
        "avg_speed_px_s": round(float(overall_avg_speed), 2),
        "traffic_density": round(sum(r["traffic_density"] for r in results if isinstance(r["lane"], int)), 3)
    })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ground Truth Metrics for Smart Toll Plaza Simulation")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="synthetic_toll_dataset",
        help="Root path to the synthetic toll dataset directory."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario ID to evaluate (1-5) or 'all'."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path for the evaluation summary (default: <dataset_dir>/metadata/evaluation_summary.csv)"
    )
    args = parser.parse_args()

    gt_dir = os.path.join(args.dataset_dir, "ground_truth")
    ann_dir = os.path.join(args.dataset_dir, "annotations")
    meta_dir = os.path.join(args.dataset_dir, "metadata")

    if not os.path.exists(gt_dir):
        print(f"Error: Ground truth directory not found: {gt_dir}")
        print("Please run `python generate_dataset.py` first.")
        return

    if args.output_csv is None:
        os.makedirs(meta_dir, exist_ok=True)
        out_csv_path = os.path.join(meta_dir, "evaluation_summary.csv")
    else:
        out_csv_path = args.output_csv

    if args.scenario.strip().lower() == "all":
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "scenario_*.csv")))
    else:
        sc_num = int(args.scenario)
        gt_files = [os.path.join(gt_dir, f"scenario_{sc_num:02d}.csv")]

    all_summaries = []
    print("\n" + "="*80)
    print(" GROUND TRUTH TRAFFIC EVALUATION REPORT")
    print("="*80)

    for gt_file in gt_files:
        if not os.path.exists(gt_file):
            print(f"File not found: {gt_file}")
            continue

        base_name = os.path.basename(gt_file)
        ann_file = os.path.join(ann_dir, base_name)
        df_summary = evaluate_scenario_ground_truth(gt_file, ann_file)

        if not df_summary.empty:
            all_summaries.append(df_summary)
            print(f"\n--- Scenario Results: {base_name} ---")
            print(df_summary.to_string(index=False))

    if all_summaries:
        combined_df = pd.concat(all_summaries, ignore_index=True)
        combined_df.to_csv(out_csv_path, index=False)
        print("\n" + "="*80)
        print(f"Evaluation summary successfully written to: {out_csv_path}")
        print("="*80)


if __name__ == "__main__":
    main()
