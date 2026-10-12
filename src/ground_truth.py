"""
Ground truth recorder module for Smart Toll Plaza dataset generation.
Records per-vehicle lifecycle ground truth, frame-level bounding box annotations
for YOLO benchmarking, and scenario metadata.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional
from .vehicle import Vehicle


class GroundTruthRecorder:
    """Collects and exports ground-truth research data across all simulated frames."""

    def __init__(self, scenario_id: int, fps: int):
        self.scenario_id = scenario_id
        self.fps = fps
        self.vehicles_registry: Dict[int, Vehicle] = {}
        self.frame_annotations: List[Dict[str, Any]] = []

    def register_vehicle(self, vehicle: Vehicle) -> None:
        """Register vehicle for lifecycle tracking."""
        self.vehicles_registry[vehicle.vehicle_id] = vehicle

    def record_frame(self, frame_idx: int, active_vehicles: List[Vehicle]) -> None:
        """
        Record frame-level vehicle detections (bounding boxes) for currently visible vehicles.
        Matches YOLO evaluation schema: frame,vehicle_id,vehicle_type,lane,x1,y1,x2,y2.
        """
        for v in active_vehicles:
            if v.is_visible_in_frame():
                x1, y1, x2, y2 = v.get_bounding_box()
                # Ensure valid box dimensions
                if x2 > x1 and y2 > y1:
                    self.frame_annotations.append({
                        "frame": frame_idx,
                        "vehicle_id": v.vehicle_id,
                        "vehicle_type": v.vehicle_type,
                        "lane": v.lane,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })

    def export_vehicle_ground_truth(self, filepath: str) -> pd.DataFrame:
        """
        Exports summary vehicle-level ground truth CSV:
        scenario_id, vehicle_id, vehicle_type, lane, entry_frame, exit_frame,
        entry_time, exit_time, max_queue_pos, stopping_duration, waiting_time
        """
        records = []
        for vid, v in sorted(self.vehicles_registry.items()):
            entry_f = v.entry_frame if v.entry_frame is not None else -1
            exit_f = v.exit_frame if v.exit_frame is not None else -1
            entry_t = round(entry_f / self.fps, 2) if entry_f >= 0 else -1.0
            exit_t = round(exit_f / self.fps, 2) if exit_f >= 0 else -1.0

            records.append({
                "scenario_id": self.scenario_id,
                "vehicle_id": v.vehicle_id,
                "vehicle_type": v.vehicle_type,
                "lane": v.lane,
                "entry_frame": entry_f,
                "exit_frame": exit_f,
                "entry_time": entry_t,
                "exit_time": exit_t,
                "max_queue_pos": v.max_queue_position,
                "stopping_duration": round(v.toll_duration, 2),
                "waiting_time": round(v.waiting_time, 2)
            })

        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        return df

    def export_frame_annotations(self, filepath: str) -> pd.DataFrame:
        """
        Exports frame-level bounding boxes CSV for YOLO detection & tracking evaluation:
        frame,vehicle_id,vehicle_type,lane,x1,y1,x2,y2
        """
        df = pd.DataFrame(self.frame_annotations)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        return df


def save_scenario_metadata(metadata_list: List[Dict[str, Any]], filepath: str) -> None:
    """Saves high-level metadata for all generated scenarios."""
    df = pd.DataFrame(metadata_list)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
