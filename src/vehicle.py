"""
Vehicle class and physical modeling for Smart Toll Plaza Simulation.
Represents individual vehicles, their physical states, visual characteristics,
and tracking history.
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List

from .config import (
    FRAME_WIDTH, FRAME_HEIGHT, ENTRY_Y_VISIBLE, EXIT_Y_VISIBLE,
    TOLL_STOP_Y, get_lane_center_x, get_perspective_scale, VEHICLE_SPECS
)


@dataclass
class Vehicle:
    """Represents a simulated vehicle travelling along a toll lane."""
    vehicle_id: int
    vehicle_type: str
    lane: int
    spawn_time: float                     # seconds from simulation start
    target_speed: float                   # cruising speed in px/s
    toll_duration: float                  # required stopping duration at toll booth (s)
    base_length: float
    base_width: float
    accel: float                          # maximum acceleration in px/s^2
    decel: float                          # maximum braking deceleration in px/s^2
    min_following_gap: float              # bumper-to-bumper safety distance in px
    color: Tuple[int, int, int] = (200, 200, 200) # Optional fallback color
    lateral_offset: float = 0.0           # slight deviation from lane center line (-4 to +4 px)
    variant_idx: int = 0                  # model/appearance variant index for sprite selection
    brightness_jitter: float = 1.0        # subtle realistic daylight intensity variation (0.95 - 1.05)
    angle_jitter: float = 0.0             # small natural heading angle deviation (-1.5 to +1.5 deg)

    # Dynamic state variables
    y: float = -40.0                      # Front bumper Y coordinate (moves 0 -> 720)
    speed: float = 0.0                    # Current speed in px/s
    acceleration: float = 0.0             # Current acceleration in px/s^2
    state: str = "spawned"                # 'spawned', 'approaching', 'queuing', 'paying', 'clearing', 'exiting', 'exited'
    current_rendered_bbox: Optional[Tuple[int, int, int, int]] = None
    
    # Toll & queue metrics
    has_paid: bool = False
    time_stopped_at_toll: float = 0.0     # Time spent paying at front of booth
    waiting_time: float = 0.0             # Total time spent delayed (speed < 5 px/s)
    max_queue_position: int = 0           # 1 = head of queue, 2 = 2nd, etc. (0 = no queue)
    
    # Ground-truth tracking frame indices
    entry_frame: Optional[int] = None
    exit_frame: Optional[int] = None
    first_stop_frame: Optional[int] = None
    departure_frame: Optional[int] = None

    # Visual state
    brake_lights: bool = False
    turn_indicator: bool = False

    def __post_init__(self):
        self.speed = self.target_speed

    @property
    def current_scale(self) -> float:
        """Perspective scaling factor at current Y position."""
        return get_perspective_scale(self.y)

    @property
    def length(self) -> float:
        """Current perspective-scaled length in pixels."""
        return self.base_length * self.current_scale

    @property
    def width(self) -> float:
        """Current perspective-scaled width in pixels."""
        return self.base_width * self.current_scale

    @property
    def rear_bumper_y(self) -> float:
        """Y coordinate of the vehicle's rear bumper."""
        return self.y - self.length

    @property
    def front_bumper_y(self) -> float:
        """Y coordinate of the vehicle's front bumper."""
        return self.y

    @property
    def center_x(self) -> float:
        """Calculates current X coordinate along the lane center line."""
        base_x = get_lane_center_x(self.lane, self.y)
        return base_x + self.lateral_offset

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """
        Returns bounding box [x1, y1, x2, y2] for YOLO detection evaluation.
        Corresponds to the actual visible vehicle sprite on screen.
        Clamped to image dimensions.
        """
        if self.current_rendered_bbox is not None:
            return self.current_rendered_bbox

        cx = self.center_x
        hw = self.width / 2.0
        
        # Vehicle travels downwards: rear is at lower Y, front bumper at higher Y
        y1 = self.rear_bumper_y
        y2 = self.front_bumper_y
        x1 = cx - hw
        x2 = cx + hw

        # Clamp to frame bounds
        x1_c = max(0, min(FRAME_WIDTH - 1, int(round(x1))))
        y1_c = max(0, min(FRAME_HEIGHT - 1, int(round(y1))))
        x2_c = max(0, min(FRAME_WIDTH - 1, int(round(x2))))
        y2_c = max(0, min(FRAME_HEIGHT - 1, int(round(y2))))

        return (x1_c, y1_c, x2_c, y2_c)

    def is_visible_in_frame(self) -> bool:
        """Returns True if any part of the vehicle is visible on camera."""
        return (self.front_bumper_y >= 0 and self.rear_bumper_y <= FRAME_HEIGHT)

    def is_in_entry_zone(self) -> bool:
        """Checks if vehicle front bumper is in the entry zone [30, 140]."""
        return 30.0 <= self.front_bumper_y <= 140.0

    def is_in_toll_zone(self) -> bool:
        """Checks if vehicle is in the toll/stop zone [400, 490]."""
        return 400.0 <= self.front_bumper_y <= 490.0

    def is_in_exit_zone(self) -> bool:
        """Checks if vehicle is in the exit zone [590, 710]."""
        return 590.0 <= self.front_bumper_y <= 710.0

    def update_physics(self, dt: float, desired_accel: float) -> None:
        """
        Integrates kinematics for a time step dt using safe acceleration limits.
        """
        # Apply deceleration or acceleration
        self.acceleration = max(-self.decel, min(self.accel, desired_accel))
        
        # Brake lights active during significant deceleration
        self.brake_lights = (self.acceleration < -15.0 or self.speed < 1.0 and self.front_bumper_y < TOLL_STOP_Y + 10)

        # Update speed (cannot drive backwards)
        new_speed = max(0.0, self.speed + self.acceleration * dt)
        
        # Update position using trapezoidal integration
        avg_speed = (self.speed + new_speed) / 2.0
        self.y += avg_speed * dt
        self.speed = new_speed

        # Track waiting / queued time if speed is near standstill
        if self.front_bumper_y >= ENTRY_Y_VISIBLE and self.rear_bumper_y <= EXIT_Y_VISIBLE:
            if self.speed < 5.0:
                self.waiting_time += dt

    def record_frame_events(self, frame_idx: int, fps: int) -> None:
        """Update entry/exit frame numbers based on position."""
        if self.entry_frame is None and self.front_bumper_y >= ENTRY_Y_VISIBLE:
            self.entry_frame = frame_idx
        
        if self.rear_bumper_y >= EXIT_Y_VISIBLE and self.exit_frame is None:
            self.exit_frame = frame_idx
            self.state = "exited"
