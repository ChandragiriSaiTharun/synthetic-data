"""
TollBooth class representing an automated or manual toll collection lane booth.
Controls the toll barrier arm animation, payment state, and overhead lane signals.
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from .config import TOLL_BOOTHS, TOLL_STOP_Y, TOLL_PASS_Y
from .vehicle import Vehicle


class TollBooth:
    """Simulates a single toll booth station for one lane."""

    def __init__(self, lane_id: int):
        self.lane_id = lane_id
        config = TOLL_BOOTHS[lane_id]
        self.cabin_center = config["cabin_center"]
        self.barrier_pivot = config["barrier_pivot"]
        self.barrier_length = config["barrier_length"]
        self.barrier_arm_end = config["barrier_arm_end"]
        self.label_pos = config["label_pos"]

        # Barrier state: angle in degrees (0 = horizontal/closed across lane, 85 = open/vertical)
        self.barrier_angle: float = 0.0
        self.target_barrier_angle: float = 0.0
        self.barrier_speed: float = 170.0  # degrees per second

        # Processing state: 'idle', 'processing', 'open', 'closing'
        self.status: str = "idle"
        self.current_vehicle: Optional[Vehicle] = None
        self.signal_color: str = "red"  # 'red' or 'green'

    @property
    def is_barrier_open(self) -> bool:
        """True if barrier is sufficiently raised for a vehicle to proceed."""
        return self.barrier_angle >= 70.0

    @property
    def is_barrier_closed(self) -> bool:
        """True if barrier is blocking the lane."""
        return self.barrier_angle <= 5.0

    def open_barrier(self) -> None:
        """Command the barrier to open."""
        self.target_barrier_angle = 85.0
        self.signal_color = "green"

    def close_barrier(self) -> None:
        """Command the barrier to close."""
        self.target_barrier_angle = 0.0
        self.signal_color = "red"

    def update(self, dt: float) -> None:
        """Smoothly animate the barrier arm towards its target angle."""
        if self.barrier_angle < self.target_barrier_angle:
            self.barrier_angle = min(self.target_barrier_angle, self.barrier_angle + self.barrier_speed * dt)
        elif self.barrier_angle > self.target_barrier_angle:
            self.barrier_angle = max(self.target_barrier_angle, self.barrier_angle - self.barrier_speed * dt)

    def process_lane_traffic(self, booth_vehicle: Optional[Vehicle], dt: float) -> None:
        """
        Manages toll transaction for the vehicle currently at or approaching the booth.
        """
        if booth_vehicle is None:
            if not self.is_barrier_closed:
                self.close_barrier()
            self.status = "idle"
            self.current_vehicle = None
            return

        self.current_vehicle = booth_vehicle

        # Check if vehicle has already paid
        if booth_vehicle.has_paid:
            # Vehicle has paid; if its rear bumper has cleared the booth, close the barrier
            if booth_vehicle.rear_bumper_y >= TOLL_PASS_Y:
                self.close_barrier()
                self.status = "closing" if not self.is_barrier_closed else "idle"
                self.current_vehicle = None
            else:
                # Still passing through the booth area; keep barrier raised
                self.open_barrier()
                self.status = "open"
            return

        # Vehicle has NOT yet paid:
        # Check if vehicle has arrived near the stop line (within approach window)
        dist_to_stop = TOLL_STOP_Y - booth_vehicle.front_bumper_y

        if -10.0 <= dist_to_stop <= 45.0 and booth_vehicle.speed < 25.0:
            # Vehicle is at the booth paying the toll
            self.status = "processing"
            booth_vehicle.time_stopped_at_toll += dt

            # Once the required stopping duration has elapsed, transaction is complete
            if booth_vehicle.time_stopped_at_toll >= booth_vehicle.toll_duration:
                booth_vehicle.has_paid = True
                self.open_barrier()
                self.status = "open"
            else:
                self.close_barrier()
        else:
            # Still decelerating on approach
            self.close_barrier()
            self.status = "idle"
