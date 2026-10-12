"""
TrafficSimulator module coordinating traffic flow, IDM physics, toll booth operations,
vehicle rendering, and ground-truth data generation.
"""

import os
import random
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from .config import (
    FRAME_WIDTH, FRAME_HEIGHT, DEFAULT_FPS, DEFAULT_DURATION,
    SCENARIO_CONFIGS, VEHICLE_SPECS, ENTRY_Y_SPAWN, ENTRY_Y_VISIBLE,
    EXIT_Y_VISIBLE, TOLL_STOP_Y
)
from .vehicle import Vehicle
from .lane import TrafficLane
from .booth import TollBooth
from .renderer import TollPlazaRenderer
from .ground_truth import GroundTruthRecorder


class TrafficSimulator:
    """Simulates a 3-lane toll plaza scenario and produces MP4 video and ground truth."""

    def __init__(
        self,
        scenario_id: int,
        duration: float = DEFAULT_DURATION,
        fps: int = DEFAULT_FPS,
        seed: int = 42,
        draw_rois: bool = False
    ):
        self.scenario_id = scenario_id
        if scenario_id not in SCENARIO_CONFIGS:
            raise ValueError(f"Unknown scenario ID {scenario_id}. Choose between 1 and 5.")
        self.scenario_cfg = SCENARIO_CONFIGS[scenario_id]
        self.duration = float(duration)
        self.fps = int(fps)
        self.total_frames = int(round(self.duration * self.fps))
        self.dt = 1.0 / self.fps
        self.seed = seed
        self.draw_rois = draw_rois

        # Setup reproducible pseudo-random generators
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Initialize Lanes, Booths, Renderer, and Ground Truth Recorder
        self.lanes: Dict[int, TrafficLane] = {i: TrafficLane(i) for i in (1, 2, 3)}
        self.booths: Dict[int, TollBooth] = {i: TollBooth(i) for i in (1, 2, 3)}
        self.renderer = TollPlazaRenderer()
        self.gt_recorder = GroundTruthRecorder(scenario_id, self.fps)

        # Pre-generate vehicle arrival schedule
        self.scheduled_vehicles: List[Vehicle] = self._generate_vehicle_schedule()

    def _generate_vehicle_schedule(self) -> List[Vehicle]:
        """Creates vehicle instances and schedules arrival times for each lane."""
        vehicles: List[Vehicle] = []
        global_veh_id = 1
        vtypes = list(VEHICLE_SPECS.keys())
        weights = [VEHICLE_SPECS[vt]["weight_in_traffic"] for vt in vtypes]

        lane_counts = self.scenario_cfg["lane_counts"]
        toll_multipliers = self.scenario_cfg.get("toll_time_multiplier", {1: 1.0, 2: 1.0, 3: 1.0})

        for lane_id in (1, 2, 3):
            count = lane_counts.get(lane_id, 10)
            if count <= 0:
                continue

            # Generate arrival times across the simulation duration
            # Spread vehicles realistically with minimum headway
            # Use beta distribution or clustered Poisson process to create natural waves
            sim_active_window = max(5.0, self.duration - 6.0) # Stop spawning 6s before end to allow clearance
            
            # Generate random sorted arrival timestamps
            raw_intervals = np.random.exponential(scale=sim_active_window / (count + 1), size=count)
            arrival_times = np.cumsum(raw_intervals)
            # Normalize to fit within active window
            arrival_times = (arrival_times / arrival_times[-1]) * sim_active_window + random.uniform(0.2, 1.5)
            arrival_times.sort()

            # Ensure minimum physical safety headway at spawn (e.g. 1.1 seconds between spawns)
            min_headway = 1.15
            for i in range(1, len(arrival_times)):
                if arrival_times[i] < arrival_times[i-1] + min_headway:
                    arrival_times[i] = arrival_times[i-1] + min_headway + random.uniform(0.1, 0.4)

            for t_arr in arrival_times:
                vtype = random.choices(vtypes, weights=weights, k=1)[0]
                spec = VEHICLE_SPECS[vtype]

                # Random variations within realistic bounds
                speed = random.uniform(spec["speed_range"][0], spec["speed_range"][1])
                toll_t = random.uniform(spec["toll_time_range"][0], spec["toll_time_range"][1]) * toll_multipliers.get(lane_id, 1.0)
                length = spec["base_length"] * random.uniform(0.95, 1.05)
                width = spec["base_width"] * random.uniform(0.95, 1.05)
                lat_jitter = random.uniform(-2.5, 2.5)
                variant_idx = random.randint(0, 10)
                brightness = random.uniform(0.96, 1.04)
                angle_jit = random.uniform(-1.2, 1.2)

                veh = Vehicle(
                    vehicle_id=global_veh_id,
                    vehicle_type=vtype,
                    lane=lane_id,
                    spawn_time=float(t_arr),
                    target_speed=speed,
                    toll_duration=toll_t,
                    base_length=length,
                    base_width=width,
                    accel=spec["accel"],
                    decel=spec["decel"],
                    min_following_gap=spec["min_following_gap"],
                    lateral_offset=lat_jitter,
                    variant_idx=variant_idx,
                    brightness_jitter=brightness,
                    angle_jitter=angle_jit
                )
                self.gt_recorder.register_vehicle(veh)
                vehicles.append(veh)
                global_veh_id += 1

        # Sort all scheduled vehicles by arrival time
        vehicles.sort(key=lambda v: v.spawn_time)
        return vehicles

    def run_simulation(
        self,
        output_video_path: str,
        export_gt_path: str,
        export_annotations_path: str,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executes the simulation loop frame by frame, writes MP4 video,
        and saves ground truth files.
        """
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        # Setup OpenCV VideoWriter with mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            float(self.fps),
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        if not writer.isOpened():
            # Fallback codec if mp4v is unavailable
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                float(self.fps),
                (FRAME_WIDTH, FRAME_HEIGHT)
            )

        if not writer.isOpened():
            raise RuntimeError(f"Failed to open OpenCV VideoWriter for {output_video_path}")

        unspawned_vehicles = list(self.scheduled_vehicles)
        active_vehicles: List[Vehicle] = []

        print(f"Starting simulation: {self.scenario_cfg['name']} ({self.total_frames} frames @ {self.fps} FPS)...")

        for frame_idx in range(self.total_frames):
            sim_time = frame_idx * self.dt

            # 1. Spawn vehicles whose time has arrived, checking for safe space at entry
            spawned_now = []
            for veh in unspawned_vehicles:
                if veh.spawn_time <= sim_time:
                    lane = self.lanes[veh.lane]
                    # Check if upstream entry point is clear of any vehicle
                    entry_clear = True
                    for active_v in lane.active_vehicles:
                        if active_v.rear_bumper_y < ENTRY_Y_SPAWN + veh.min_following_gap:
                            entry_clear = False
                            break

                    if entry_clear:
                        veh.y = ENTRY_Y_SPAWN
                        veh.state = "approaching"
                        lane.add_vehicle(veh)
                        active_vehicles.append(veh)
                        spawned_now.append(veh)

            for s in spawned_now:
                unspawned_vehicles.remove(s)

            # 2. Process Toll Booths and Lanes
            for lane_id in (1, 2, 3):
                lane = self.lanes[lane_id]
                booth = self.booths[lane_id]

                # Update barrier angle
                booth.update(self.dt)

                # Process transaction for vehicle at booth
                booth_v = lane.get_vehicle_at_booth()
                booth.process_lane_traffic(booth_v, self.dt)

                # Update IDM car-following physics for lane
                lane.update_lane_physics(self.dt, booth)

            # 3. Check for vehicles that exited the camera frame
            exited_now = []
            for veh in active_vehicles:
                veh.record_frame_events(frame_idx, self.fps)
                if veh.rear_bumper_y >= EXIT_Y_VISIBLE:
                    lane = self.lanes[veh.lane]
                    lane.remove_vehicle(veh)
                    exited_now.append(veh)

            for ev in exited_now:
                active_vehicles.remove(ev)

            # 4. Record frame-level annotations for YOLO tracking
            self.gt_recorder.record_frame(frame_idx, active_vehicles)

            # 5. Render frame
            rendered_frame = self.renderer.render_frame(
                vehicles=active_vehicles,
                booths=self.booths,
                frame_idx=frame_idx,
                fps=self.fps,
                scenario_name=self.scenario_cfg["name"],
                draw_rois=self.draw_rois
            )

            # Write frame to video
            writer.write(rendered_frame)

            if progress_callback and frame_idx % 30 == 0:
                progress_callback(frame_idx, self.total_frames)

        writer.release()
        print(f"Video saved to: {output_video_path}")

        # Export Ground Truth Files
        df_gt = self.gt_recorder.export_vehicle_ground_truth(export_gt_path)
        print(f"Vehicle ground truth saved to: {export_gt_path} ({len(df_gt)} vehicles recorded)")

        df_ann = self.gt_recorder.export_frame_annotations(export_annotations_path)
        print(f"Frame annotations saved to: {export_annotations_path} ({len(df_ann)} bounding boxes recorded)")

        # Compile run summary
        summary = {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_cfg["name"],
            "total_vehicles_scheduled": len(self.scheduled_vehicles),
            "total_vehicles_spawned": len(self.scheduled_vehicles) - len(unspawned_vehicles),
            "total_vehicles_completed": sum(1 for v in self.gt_recorder.vehicles_registry.values() if v.exit_frame is not None),
            "max_queue_per_lane": {lane_id: self.lanes[lane_id].max_queue_size_recorded for lane_id in (1, 2, 3)},
            "video_path": output_video_path,
            "ground_truth_path": export_gt_path,
            "annotations_path": export_annotations_path
        }
        return summary
