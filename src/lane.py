"""
TrafficLane module managing vehicle movement, queue dynamics, and car-following physics
for a single toll plaza lane.
"""

from typing import List, Optional, Tuple, Dict, Any
from .config import TOLL_STOP_Y, TOLL_PASS_Y, FRAME_HEIGHT, ENTRY_Y_VISIBLE, EXIT_Y_VISIBLE
from .vehicle import Vehicle
from .booth import TollBooth


class TrafficLane:
    """Manages the vehicles and traffic dynamics inside a single lane."""

    def __init__(self, lane_id: int):
        self.lane_id = lane_id
        self.active_vehicles: List[Vehicle] = []
        self.max_queue_size_recorded: int = 0
        self.current_queue_size: int = 0

    def add_vehicle(self, vehicle: Vehicle) -> None:
        """Add newly spawned vehicle to lane."""
        self.active_vehicles.append(vehicle)

    def remove_vehicle(self, vehicle: Vehicle) -> None:
        """Remove exited vehicle from active tracking."""
        if vehicle in self.active_vehicles:
            self.active_vehicles.remove(vehicle)

    def get_lead_vehicle(self) -> Optional[Vehicle]:
        """Returns the vehicle furthest along the road (highest Y)."""
        if not self.active_vehicles:
            return None
        return max(self.active_vehicles, key=lambda v: v.y)

    def get_vehicle_at_booth(self) -> Optional[Vehicle]:
        """
        Returns the foremost vehicle that is either at the booth paying,
        or approaching the booth, but has not yet completed passage past the booth.
        """
        candidates = [v for v in self.active_vehicles if v.rear_bumper_y <= TOLL_PASS_Y]
        if not candidates:
            return None
        return max(candidates, key=lambda v: v.y)

    def compute_acceleration(
        self,
        vehicle: Vehicle,
        leader: Optional[Vehicle],
        booth: TollBooth
    ) -> float:
        """
        Calculates the desired acceleration using the Intelligent Driver Model (IDM).
        Handles both physical lead vehicles and virtual stop-line barriers.
        """
        v = max(0.0, vehicle.speed)
        v0 = vehicle.target_speed
        a_max = vehicle.accel
        b_comf = vehicle.decel
        s0 = vehicle.min_following_gap
        T = 0.9  # Safe time headway in seconds

        # Free road acceleration term
        ratio = (v / v0) if v0 > 0 else 1.0
        accel_free = a_max * (1.0 - (ratio ** 4))

        # Check interaction with physical leader ahead in lane
        interaction_term = 0.0

        if leader is not None:
            # Distance from this vehicle's front bumper to leader's rear bumper
            s_actual = leader.rear_bumper_y - vehicle.front_bumper_y
            delta_v = v - leader.speed

            # Safety margin
            s_star = s0 + v * T + (v * delta_v) / (2.0 * ((a_max * b_comf) ** 0.5) + 1e-5)
            s_star = max(s0, s_star)

            if s_actual > 0.1:
                interaction_term = a_max * ((s_star / s_actual) ** 2)
            else:
                # Emergency braking if too close
                return -b_comf

        # Check interaction with toll stop line / barrier
        # If vehicle hasn't paid, or barrier is not yet open, treat stop line as a stationary obstacle
        if (not vehicle.has_paid or not booth.is_barrier_open) and vehicle.front_bumper_y <= TOLL_STOP_Y + 10.0:
            s_toll = TOLL_STOP_Y - vehicle.front_bumper_y
            if s_toll > -5.0:
                s_star_toll = s0 + v * T + (v * v) / (2.0 * ((a_max * b_comf) ** 0.5) + 1e-5)
                s_star_toll = max(s0, s_star_toll)
                if s_toll > 2.0:
                    toll_interaction = a_max * ((s_star_toll / s_toll) ** 2)
                    interaction_term = max(interaction_term, toll_interaction)
                else:
                    return -b_comf

        desired_accel = accel_free - interaction_term
        return desired_accel

    def update_lane_physics(self, dt: float, booth: TollBooth) -> None:
        """
        Updates physics for all vehicles in this lane from furthest downstream to entry.
        Also prevents overlap and updates queuing statistics.
        """
        # Sort vehicles by front bumper Y descending (first vehicle on road is index 0)
        self.active_vehicles.sort(key=lambda v: v.y, reverse=True)

        for i, vehicle in enumerate(self.active_vehicles):
            leader = self.active_vehicles[i - 1] if i > 0 else None
            desired_accel = self.compute_acceleration(vehicle, leader, booth)
            vehicle.update_physics(dt, desired_accel)

            # Strict anti-overlap guarantee with leader
            if leader is not None:
                max_allowed_y = leader.rear_bumper_y - vehicle.min_following_gap
                if vehicle.y > max_allowed_y:
                    vehicle.y = max_allowed_y
                    vehicle.speed = min(vehicle.speed, max(0.0, leader.speed))

            # Stop bar enforcement before toll payment
            if not vehicle.has_paid and vehicle.y > TOLL_STOP_Y:
                vehicle.y = TOLL_STOP_Y
                vehicle.speed = 0.0

        # Update queue tracking
        self._update_queue_status(booth)

    def _update_queue_status(self, booth: TollBooth) -> None:
        """
        Determines which vehicles are currently part of a queue at the toll booth.
        A vehicle is in queue if it is stopped (speed < 5 px/s) upstream of the toll booth
        or connected to stopped vehicles upstream of the booth.
        """
        current_queue = 0
        in_queue_chain = False

        for i, vehicle in enumerate(self.active_vehicles):
            # Only consider vehicles between entry line and toll stop line
            if vehicle.front_bumper_y < ENTRY_Y_VISIBLE:
                continue

            if vehicle.front_bumper_y <= TOLL_STOP_Y + 15.0:
                is_near_stop = (abs(TOLL_STOP_Y - vehicle.front_bumper_y) < 25.0 and vehicle.speed < 5.0)
                is_slow = vehicle.speed < 5.0

                if i == 0:
                    # Lead vehicle at toll booth
                    if is_near_stop or (not vehicle.has_paid and is_slow):
                        current_queue += 1
                        in_queue_chain = True
                        vehicle.max_queue_position = max(vehicle.max_queue_position, current_queue)
                    else:
                        in_queue_chain = False
                else:
                    if in_queue_chain and is_slow:
                        current_queue += 1
                        vehicle.max_queue_position = max(vehicle.max_queue_position, current_queue)
                    else:
                        in_queue_chain = False
            else:
                in_queue_chain = False

        self.current_queue_size = current_queue
        self.max_queue_size_recorded = max(self.max_queue_size_recorded, current_queue)

    def get_traffic_density(self) -> float:
        """
        Calculates instantaneous traffic density:
        Fraction of visible lane occupied by vehicles (0.0 to 1.0).
        """
        visible_length = EXIT_Y_VISIBLE - ENTRY_Y_VISIBLE
        total_veh_length = sum(v.length for v in self.active_vehicles if v.is_visible_in_frame())
        return min(1.0, total_veh_length / visible_length) if visible_length > 0 else 0.0
