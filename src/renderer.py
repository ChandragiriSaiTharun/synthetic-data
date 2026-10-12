"""
Renderer module for the Smart Toll Plaza Simulator.
Renders realistic CCTV-style imagery of the toll plaza road, lane markings,
concrete divider islands, toll booth cabins, dynamic barrier arms, overhead signage,
and detailed vehicle models with drop shadows and brake lights.
"""

import math
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .config import (
    FRAME_WIDTH, FRAME_HEIGHT, LANE_POLYGONS, ISLAND_POLYGONS,
    TOLL_BOOTHS, GLOBAL_ROIS, TOLL_STOP_Y, get_lane_rois, get_lane_angle
)
from .vehicle import Vehicle
from .booth import TollBooth
from .asset_manager import VehicleAssetManager


class TollPlazaRenderer:
    """Renders the complete toll plaza environment and moving vehicles."""

    def __init__(self, asset_dir: str = "assets/vehicles"):
        self.asset_manager = VehicleAssetManager(asset_dir)
        # Pre-render the static background for performance and visual consistency
        self.static_background = self._generate_static_background()

    def _generate_static_background(self) -> np.ndarray:
        """Constructs the high-fidelity static background image."""
        bg = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        # 1. Outer grass / roadside embankments (subtle muted green-grey)
        bg[:] = (75, 82, 70)  # BGR

        # 2. Main paved plaza apron (asphalt surface)
        # Bounded by left and right outer shoulders
        asphalt_poly = np.array([
            [260, 0], [930, 0],
            [1060, FRAME_HEIGHT], [190, FRAME_HEIGHT]
        ], dtype=np.int32)
        cv2.fillPoly(bg, [asphalt_poly], (52, 50, 48))

        # Add subtle asphalt texture noise
        noise = np.random.RandomState(42).normal(0, 3, bg.shape).astype(np.int16)
        bg_int = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        bg = bg_int

        # Subtle tire friction tracks along lane centers
        for lane_id, poly in LANE_POLYGONS.items():
            tl, tr, br, bl = poly
            cx_top = (tl[0] + tr[0]) // 2
            cx_bot = (bl[0] + br[0]) // 2
            for offset in (-18, 18):
                pt1 = (cx_top + offset, 0)
                pt2 = (cx_bot + offset, FRAME_HEIGHT)
                cv2.line(bg, pt1, pt2, (44, 42, 40), 12, cv2.LINE_AA)

        # 3. Paint Lane Road Markings
        self._draw_lane_markings(bg)

        # 4. Concrete Toll Booth Islands (raised curbs)
        self._draw_concrete_islands(bg)

        # 5. Painted Speed Markings and Direction Arrows
        self._draw_road_markings(bg)

        # 6. Toll Booth Buildings (base structures)
        self._draw_toll_booth_structures(bg)

        return bg

    def _draw_lane_markings(self, img: np.ndarray) -> None:
        """Paints road surface lines, solid white borders, and stop lines."""
        # Lane divider dashed lines upstream of toll plaza (y = 0 to 360)
        # Line between Lane 1 and Lane 2
        for y in range(20, 340, 40):
            a1 = y / FRAME_HEIGHT
            a2 = (y + 22) / FRAME_HEIGHT
            # L1-L2 boundary
            x1 = int(480 + (460 - 480) * a1)
            x2 = int(480 + (460 - 480) * a2)
            cv2.line(img, (x1, y), (x2, y + 22), (220, 220, 220), 3, cv2.LINE_AA)
            # L2-L3 boundary
            x1_3 = int(680 + (720 - 680) * a1)
            x2_3 = int(680 + (720 - 680) * a2)
            cv2.line(img, (x1_3, y), (x2_3, y + 22), (220, 220, 220), 3, cv2.LINE_AA)

        # Solid white stop bars at toll booths (y = 450)
        for lane_id, poly in LANE_POLYGONS.items():
            tl, tr, br, bl = poly
            a = TOLL_STOP_Y / FRAME_HEIGHT
            xl = int(tl[0] + (bl[0] - tl[0]) * a)
            xr = int(tr[0] + (br[0] - tr[0]) * a)
            # Broad retro-reflective white stop bar
            cv2.line(img, (xl + 8, int(TOLL_STOP_Y)), (xr - 8, int(TOLL_STOP_Y)), (240, 240, 245), 7, cv2.LINE_AA)
            # Secondary warning line
            cv2.line(img, (xl + 8, int(TOLL_STOP_Y) - 14), (xr - 8, int(TOLL_STOP_Y) - 14), (200, 200, 205), 2, cv2.LINE_AA)

    def _draw_concrete_islands(self, img: np.ndarray) -> None:
        """Renders raised concrete divider islands with striped safety curbs."""
        for name, poly in ISLAND_POLYGONS.items():
            pts = np.array(poly, dtype=np.int32)
            # Base concrete slab
            cv2.fillPoly(img, [pts], (165, 168, 172))
            # Dark curb shadow line
            cv2.polylines(img, [pts], True, (115, 118, 122), 2, cv2.LINE_AA)

            # Yellow and black diagonal hazard warning stripes at island noses (y: 280-360 and 540-620)
            for y_start, y_end in [(280, 360), (540, 620)]:
                for y_stripe in range(y_start, y_end, 16):
                    a = y_stripe / FRAME_HEIGHT
                    # Find left and right of island
                    tl, tr, br, bl = poly
                    xl = int(tl[0] + (bl[0] - tl[0]) * a)
                    xr = int(tr[0] + (br[0] - tr[0]) * a)
                    cv2.line(img, (xl, y_stripe), (xr, y_stripe + 10), (35, 185, 220), 4, cv2.LINE_AA)

    def _draw_road_markings(self, img: np.ndarray) -> None:
        """Draws directional lane arrows and speed limit numerals on pavement."""
        # White forward-pointing traffic arrows in each lane upstream
        for lane_id, poly in LANE_POLYGONS.items():
            tl, tr, br, bl = poly
            # Upstream arrow at y = 180
            a = 180 / FRAME_HEIGHT
            cx = int((tl[0] + tr[0]) / 2 + ((bl[0] + br[0]) / 2 - (tl[0] + tr[0]) / 2) * a)
            self._draw_pavement_arrow(img, cx, 180)

            # Downstream arrow at y = 620
            a2 = 620 / FRAME_HEIGHT
            cx2 = int((tl[0] + tr[0]) / 2 + ((bl[0] + br[0]) / 2 - (tl[0] + tr[0]) / 2) * a2)
            self._draw_pavement_arrow(img, cx2, 620)

    def _draw_pavement_arrow(self, img: np.ndarray, cx: int, cy: int) -> None:
        """Draws an elongated white CCTV road directional arrow."""
        color = (225, 225, 230)
        # Shaft
        cv2.line(img, (cx, cy - 25), (cx, cy + 25), color, 4, cv2.LINE_AA)
        # Arrowhead
        arrow_head = np.array([
            [cx, cy + 30],
            [cx - 12, cy + 12],
            [cx + 12, cy + 12]
        ], dtype=np.int32)
        cv2.fillPoly(img, [arrow_head], color)

    def _draw_toll_booth_structures(self, img: np.ndarray) -> None:
        """Draws toll booth cabins situated on the divider islands."""
        for lane_id, booth in TOLL_BOOTHS.items():
            cx, cy = booth["cabin_center"]
            w, h = 32, 70  # Width and height of booth cabin

            # Cabin shadow
            cv2.rectangle(img, (cx - w//2 + 6, cy - h//2 + 6), (cx + w//2 + 6, cy + h//2 + 6), (40, 40, 40), -1)

            # Cabin main body (grey-blue modern toll booth)
            cv2.rectangle(img, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), (180, 185, 190), -1)
            cv2.rectangle(img, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), (90, 95, 100), 2)

            # Tinted glass windows
            cv2.rectangle(img, (cx - w//2 + 4, cy - h//2 + 8), (cx + w//2 - 4, cy - h//2 + 30), (120, 100, 70), -1)
            cv2.rectangle(img, (cx - w//2 + 4, cy - h//2 + 8), (cx + w//2 - 4, cy - h//2 + 30), (50, 50, 50), 1)

            # Booth Roof Unit / AC cover
            cv2.rectangle(img, (cx - w//2 + 2, cy - h//2 + 2), (cx + w//2 - 2, cy - h//2 + 6), (220, 220, 225), -1)

            # Safety Barrier Motor Housing (bright yellow/orange pillar)
            px, py = booth["barrier_pivot"]
            cv2.rectangle(img, (px - 6, py - 8), (px + 6, py + 8), (20, 140, 220), -1)
            cv2.rectangle(img, (px - 6, py - 8), (px + 6, py + 8), (10, 80, 140), 1)
            cv2.circle(img, (px, py), 3, (40, 40, 40), -1)

    def _draw_overhead_gantry(self, img: np.ndarray) -> None:
        """Renders the steel truss gantry spanning the 3 lanes."""
        y_gantry = 390
        # Steel truss crossbeam
        cv2.line(img, (240, y_gantry), (980, y_gantry), (100, 105, 110), 10, cv2.LINE_AA)
        cv2.line(img, (240, y_gantry + 8), (980, y_gantry + 8), (70, 75, 80), 3, cv2.LINE_AA)
        
        # Support vertical pillars on left and right shoulders
        cv2.rectangle(img, (230, y_gantry - 15), (245, y_gantry + 30), (80, 85, 90), -1)
        cv2.rectangle(img, (975, y_gantry - 15), (990, y_gantry + 30), (80, 85, 90), -1)

    def render_frame(
        self,
        vehicles: List[Vehicle],
        booths: Dict[int, TollBooth],
        frame_idx: int,
        fps: int,
        scenario_name: str,
        draw_rois: bool = False
    ) -> np.ndarray:
        """
        Renders an entire frame including static background, dynamic barrier arms,
        moving vehicles with shadows, overhead electronic signs, and CCTV overlay.
        """
        # Start with a copy of static background
        frame = self.static_background.copy()

        # 1. Optionally draw ROI boundaries if requested
        if draw_rois:
            self._draw_roi_overlays(frame)

        # 2. Draw moving vehicles (sorted by Y so downstream vehicles appear in front)
        visible_vehicles = [v for v in vehicles if v.is_visible_in_frame()]
        # Sort ascending by Y so vehicles further up (smaller Y) are drawn before vehicles closer (larger Y)
        visible_vehicles.sort(key=lambda v: v.y)

        for vehicle in visible_vehicles:
            self._draw_vehicle(frame, vehicle)

        # 3. Draw overhead gantry truss (vehicles drive under the gantry)
        self._draw_overhead_gantry(frame)

        # 4. Draw dynamic toll barrier arms and overhead lane signage
        for lane_id, booth in booths.items():
            self._draw_barrier_arm(frame, booth)
            self._draw_overhead_lane_sign(frame, booth)

        # 4. CCTV On-Screen Display (OSD) overlay
        self._draw_cctv_osd(frame, frame_idx, fps, scenario_name)

        return frame

    def _draw_barrier_arm(self, img: np.ndarray, booth: TollBooth) -> None:
        """Draws the rotating red-and-white striped barrier arm with ground shadow."""
        px, py = booth.barrier_pivot
        ex, ey = booth.barrier_arm_end
        full_length = abs(ex - px)

        # Angle: 0 deg = horizontal across lane; 85 deg = raised vertical
        angle_rad = math.radians(booth.barrier_angle)
        
        # Determine direction of arm based on which side pivot is on
        # For Lane 1 & 2 pivot is on right side of lane, arm extends left (negative dx)
        # For Lane 3 pivot is on left side of lane, arm extends right (positive dx)
        dx_dir = -1.0 if ex < px else 1.0

        cur_length = full_length * math.cos(angle_rad)
        tip_x = int(px + dx_dir * cur_length)
        tip_y = int(py - full_length * math.sin(angle_rad) * 0.45) # perspective foreshortening

        # Ground shadow of barrier
        shadow_tip_x = int(px + dx_dir * cur_length * 0.95 + 6)
        cv2.line(img, (px + 6, py + 6), (shadow_tip_x, py + 6), (40, 40, 40), 4, cv2.LINE_AA)

        # Striped barrier arm
        # Draw base white bar
        cv2.line(img, (px, py), (tip_x, tip_y), (245, 245, 250), 5, cv2.LINE_AA)

        # Draw red stripes along arm
        num_stripes = 6
        for s in range(num_stripes):
            t1 = s / num_stripes
            t2 = (s + 0.45) / num_stripes
            p1 = (int(px + (tip_x - px) * t1), int(py + (tip_y - py) * t1))
            p2 = (int(px + (tip_x - px) * t2), int(py + (tip_y - py) * t2))
            cv2.line(img, p1, p2, (30, 40, 220), 5, cv2.LINE_AA)

        # Pivot nut
        cv2.circle(img, (px, py), 4, (30, 30, 30), -1)

    def _draw_overhead_lane_sign(self, img: np.ndarray, booth: TollBooth) -> None:
        """Renders electronic signboard above each lane showing Lane number and Signal."""
        lx, ly = booth.label_pos
        w_sign, h_sign = 84, 34

        # Signboard housing (dark metallic case)
        cv2.rectangle(img, (lx - w_sign//2, ly - h_sign//2), (lx + w_sign//2, ly + h_sign//2), (25, 28, 30), -1)
        cv2.rectangle(img, (lx - w_sign//2, ly - h_sign//2), (lx + w_sign//2, ly + h_sign//2), (180, 185, 190), 2)

        # Text: "LANE 1", etc.
        lane_text = f"LANE {booth.lane_id}"
        cv2.putText(img, lane_text, (lx - 34, ly - 3), cv2.FONT_HERSHEY_DUPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

        # Dynamic Electronic Signal:
        # Green Downward Arrow (OPEN) or Red Cross / Stop Bar (CLOSED)
        sig_x = lx
        sig_y = ly + 10
        if booth.signal_color == "green" or booth.is_barrier_open:
            # Green Arrow
            cv2.circle(img, (sig_x, sig_y), 4, (40, 230, 80), -1)
            cv2.line(img, (sig_x, sig_y - 3), (sig_x, sig_y + 3), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(img, (sig_x - 3, sig_y), (sig_x, sig_y + 3), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(img, (sig_x + 3, sig_y), (sig_x, sig_y + 3), (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # Red Cross
            cv2.circle(img, (sig_x, sig_y), 4, (40, 40, 235), -1)
            cv2.line(img, (sig_x - 3, sig_y - 3), (sig_x + 3, sig_y + 3), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(img, (sig_x + 3, sig_y - 3), (sig_x - 3, sig_y + 3), (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_vehicle(self, img: np.ndarray, vehicle: Vehicle) -> None:
        """Renders realistic vehicle sprite asset with perspective scaling, rotation, and soft shadow."""
        cx = vehicle.center_x
        cy = (vehicle.front_bumper_y + vehicle.rear_bumper_y) / 2.0
        w = vehicle.width
        length = vehicle.length

        # Lane trajectory perspective angle + natural slight heading deviation
        lane_angle = get_lane_angle(vehicle.lane, cy) + vehicle.angle_jitter

        # Retrieve transparent RGBA vehicle sprite asset
        sprite_bgra, sprite_name = self.asset_manager.get_sprite(vehicle.vehicle_type, vehicle.variant_idx)

        # Composite vehicle sprite with natural soft contact shadow and perspective scale
        bbox = self.asset_manager.render_vehicle_sprite(
            canvas=img,
            sprite_bgra=sprite_bgra,
            cx=cx,
            cy=cy,
            target_w=w,
            target_h=length,
            angle_deg=lane_angle,
            brightness_jitter=vehicle.brightness_jitter
        )
        vehicle.current_rendered_bbox = bbox

        # Taillights / glowing brake lights when decelerating or queued
        if vehicle.brake_lights:
            hw = w / 2.0
            yr = int(round(vehicle.rear_bumper_y))
            # Left and right brake light points
            lx = int(round(cx - hw * 0.65))
            rx = int(round(cx + hw * 0.65))
            ly = yr + 4
            if 0 <= ly < FRAME_HEIGHT:
                if 0 <= lx < FRAME_WIDTH:
                    cv2.circle(img, (lx, ly), 5, (20, 20, 255), -1)
                    cv2.circle(img, (lx, ly), 2, (180, 180, 255), -1)
                if 0 <= rx < FRAME_WIDTH:
                    cv2.circle(img, (rx, ly), 5, (20, 20, 255), -1)
                    cv2.circle(img, (rx, ly), 2, (180, 180, 255), -1)

    def _draw_car_or_suv(
        self,
        img: np.ndarray,
        cx: float,
        yf: float,
        yr: float,
        w: float,
        length: float,
        veh: Vehicle,
        is_suv: bool
    ) -> None:
        """Renders realistic passenger car or SUV."""
        hw = w / 2.0
        color = veh.color
        dark_color = tuple(max(0, c - 45) for c in color)
        light_color = tuple(min(255, c + 35) for c in color)

        # Rounded vehicle outer body polygon
        body_pts = np.array([
            [int(cx - hw + 2), int(yr + 3)],
            [int(cx + hw - 2), int(yr + 3)],
            [int(cx + hw), int(yr + 12)],
            [int(cx + hw), int(yf - 10)],
            [int(cx + hw - 4), int(yf)],
            [int(cx - hw + 4), int(yf)],
            [int(cx - hw), int(yf - 10)],
            [int(cx - hw), int(yr + 12)],
        ], dtype=np.int32)

        cv2.fillPoly(img, [body_pts], color)
        cv2.polylines(img, [body_pts], True, dark_color, 1, cv2.LINE_AA)

        # Hood / Front windshield / Roof / Rear windshield
        hood_len = length * (0.24 if not is_suv else 0.22)
        roof_len = length * (0.42 if not is_suv else 0.52)
        trunk_len = length * (0.20 if not is_suv else 0.12)

        y_hood_front = yf - 2
        y_hood_back = yf - hood_len
        y_roof_back = y_hood_back - roof_len
        y_trunk_back = yr + 2

        # Windshield (dark reflective tint)
        windshield = np.array([
            [int(cx - hw + 4), int(y_hood_back)],
            [int(cx + hw - 4), int(y_hood_back)],
            [int(cx + hw - 6), int(y_hood_back - 12)],
            [int(cx - hw + 6), int(y_hood_back - 12)],
        ], dtype=np.int32)
        cv2.fillPoly(img, [windshield], (45, 50, 55))

        # Roof
        roof_pts = np.array([
            [int(cx - hw + 6), int(y_hood_back - 12)],
            [int(cx + hw - 6), int(y_hood_back - 12)],
            [int(cx + hw - 5), int(y_roof_back + 8)],
            [int(cx - hw + 5), int(y_roof_back + 8)],
        ], dtype=np.int32)
        cv2.fillPoly(img, [roof_pts], light_color if not is_suv else dark_color)

        if is_suv:
            # Roof rack rails
            cv2.line(img, (int(cx - hw + 8), int(y_hood_back - 10)), (int(cx - hw + 8), int(y_roof_back + 10)), (30, 30, 30), 2)
            cv2.line(img, (int(cx + hw - 8), int(y_hood_back - 10)), (int(cx + hw - 8), int(y_roof_back + 10)), (30, 30, 30), 2)

        # Rear windshield
        rear_glass = np.array([
            [int(cx - hw + 5), int(y_roof_back + 8)],
            [int(cx + hw - 5), int(y_roof_back + 8)],
            [int(cx + hw - 4), int(y_roof_back)],
            [int(cx - hw + 4), int(y_roof_back)],
        ], dtype=np.int32)
        cv2.fillPoly(img, [rear_glass], (45, 50, 55))

        # Side mirrors
        cv2.line(img, (int(cx - hw), int(y_hood_back - 4)), (int(cx - hw - 3), int(y_hood_back - 6)), dark_color, 2)
        cv2.line(img, (int(cx + hw), int(y_hood_back - 4)), (int(cx + hw + 3), int(y_hood_back - 6)), dark_color, 2)

        # Headlights (warm white/yellow)
        cv2.circle(img, (int(cx - hw + 5), int(yf - 2)), 2, (210, 245, 255), -1)
        cv2.circle(img, (int(cx + hw - 5), int(yf - 2)), 2, (210, 245, 255), -1)

        # Taillights (standard red, or intense glowing red if braking!)
        if veh.brake_lights:
            # Bright glowing brake lights with halo
            cv2.circle(img, (int(cx - hw + 4), int(yr + 3)), 5, (20, 20, 255), -1)
            cv2.circle(img, (int(cx + hw - 4), int(yr + 3)), 5, (20, 20, 255), -1)
            cv2.circle(img, (int(cx - hw + 4), int(yr + 3)), 2, (180, 180, 255), -1)
            cv2.circle(img, (int(cx + hw - 4), int(yr + 3)), 2, (180, 180, 255), -1)
        else:
            cv2.rectangle(img, (int(cx - hw + 2), int(yr + 1)), (int(cx - hw + 6), int(yr + 4)), (20, 25, 180), -1)
            cv2.rectangle(img, (int(cx + hw - 6), int(yr + 1)), (int(cx + hw - 2), int(yr + 4)), (20, 25, 180), -1)

    def _draw_truck(
        self,
        img: np.ndarray,
        cx: float,
        yf: float,
        yr: float,
        w: float,
        length: float,
        veh: Vehicle
    ) -> None:
        """Renders freight truck with distinct cab and cargo trailer."""
        hw = w / 2.0
        color = veh.color
        dark_color = tuple(max(0, c - 40) for c in color)

        cab_len = length * 0.28
        trailer_len = length * 0.68
        gap = length * 0.04

        y_cab_front = yf
        y_cab_back = yf - cab_len
        y_trailer_front = y_cab_back - gap
        y_trailer_back = yr

        # 1. Cargo Container / Trailer (rear section)
        trailer_box = np.array([
            [int(cx - hw), int(y_trailer_back)],
            [int(cx + hw), int(y_trailer_back)],
            [int(cx + hw), int(y_trailer_front)],
            [int(cx - hw), int(y_trailer_front)],
        ], dtype=np.int32)
        cv2.fillPoly(img, [trailer_box], (210, 215, 220))
        cv2.polylines(img, [trailer_box], True, (130, 135, 140), 2)

        # Corrugated container roof ribs
        for y_rib in range(int(y_trailer_back + 10), int(y_trailer_front - 6), 14):
            cv2.line(img, (int(cx - hw + 4), y_rib), (int(cx + hw - 4), y_rib), (180, 185, 190), 1)

        # 2. Truck Cab (front section)
        cab_box = np.array([
            [int(cx - hw + 2), int(y_cab_back)],
            [int(cx + hw - 2), int(y_cab_back)],
            [int(cx + hw - 1), int(y_cab_front - 6)],
            [int(cx + hw - 4), int(y_cab_front)],
            [int(cx - hw + 4), int(y_cab_front)],
            [int(cx - hw + 1), int(y_cab_front - 6)],
        ], dtype=np.int32)
        cv2.fillPoly(img, [cab_box], color)
        cv2.polylines(img, [cab_box], True, dark_color, 1)

        # Windshield
        cv2.rectangle(img, (int(cx - hw + 4), int(y_cab_front - 14)), (int(cx + hw - 4), int(y_cab_front - 6)), (45, 50, 55), -1)

        # Headlights & Taillights
        cv2.circle(img, (int(cx - hw + 4), int(yf - 2)), 3, (210, 245, 255), -1)
        cv2.circle(img, (int(cx + hw - 4), int(yf - 2)), 3, (210, 245, 255), -1)

        if veh.brake_lights:
            cv2.circle(img, (int(cx - hw + 3), int(yr + 3)), 5, (20, 20, 255), -1)
            cv2.circle(img, (int(cx + hw - 3), int(yr + 3)), 5, (20, 20, 255), -1)
        else:
            cv2.rectangle(img, (int(cx - hw + 2), int(yr + 1)), (int(cx - hw + 6), int(yr + 4)), (20, 25, 180), -1)
            cv2.rectangle(img, (int(cx + hw - 6), int(yr + 1)), (int(cx + hw - 2), int(yr + 4)), (20, 25, 180), -1)

    def _draw_bus(
        self,
        img: np.ndarray,
        cx: float,
        yf: float,
        yr: float,
        w: float,
        length: float,
        veh: Vehicle
    ) -> None:
        """Renders passenger bus or coach."""
        hw = w / 2.0
        color = veh.color
        dark_color = tuple(max(0, c - 40) for c in color)

        # Aerodynamic curved body
        body = np.array([
            [int(cx - hw + 3), int(yr + 2)],
            [int(cx + hw - 3), int(yr + 2)],
            [int(cx + hw), int(yr + 8)],
            [int(cx + hw), int(yf - 10)],
            [int(cx + hw - 3), int(yf)],
            [int(cx - hw + 3), int(yf)],
            [int(cx - hw), int(yf - 10)],
            [int(cx - hw), int(yr + 8)],
        ], dtype=np.int32)
        cv2.fillPoly(img, [body], color)
        cv2.polylines(img, [body], True, dark_color, 1)

        # Front panoramic windshield
        cv2.rectangle(img, (int(cx - hw + 4), int(yf - 14)), (int(cx + hw - 4), int(yf - 4)), (45, 50, 55), -1)

        # Roof AC pods
        cv2.rectangle(img, (int(cx - 8), int((yf + yr) / 2 - 18)), (int(cx + 8), int((yf + yr) / 2 + 18)), (230, 230, 235), -1)
        cv2.rectangle(img, (int(cx - 8), int((yf + yr) / 2 - 18)), (int(cx + 8), int((yf + yr) / 2 + 18)), (140, 140, 145), 1)

        # Headlights & Taillights
        cv2.circle(img, (int(cx - hw + 4), int(yf - 2)), 3, (210, 245, 255), -1)
        cv2.circle(img, (int(cx + hw - 4), int(yf - 2)), 3, (210, 245, 255), -1)

        if veh.brake_lights:
            cv2.circle(img, (int(cx - hw + 3), int(yr + 3)), 5, (20, 20, 255), -1)
            cv2.circle(img, (int(cx + hw - 3), int(yr + 3)), 5, (20, 20, 255), -1)
        else:
            cv2.rectangle(img, (int(cx - hw + 2), int(yr + 1)), (int(cx - hw + 6), int(yr + 4)), (20, 25, 180), -1)
            cv2.rectangle(img, (int(cx + hw - 6), int(yr + 1)), (int(cx + hw - 2), int(yr + 4)), (20, 25, 180), -1)

    def _draw_motorcycle(
        self,
        img: np.ndarray,
        cx: float,
        yf: float,
        yr: float,
        w: float,
        length: float,
        veh: Vehicle
    ) -> None:
        """Renders motorcycle and rider helmet."""
        color = veh.color
        # Slim bike chassis
        cv2.line(img, (int(cx), int(yr + 2)), (int(cx), int(yf - 2)), color, 5, cv2.LINE_AA)
        
        # Handlebars
        cv2.line(img, (int(cx - 7), int(yf - 7)), (int(cx + 7), int(yf - 7)), (40, 40, 40), 2, cv2.LINE_AA)

        # Rider helmet
        helmet_y = int(yf - length * 0.55)
        cv2.circle(img, (int(cx), helmet_y), 4, (35, 40, 45), -1)

        # Headlight & Taillight
        cv2.circle(img, (int(cx), int(yf - 1)), 2, (220, 250, 255), -1)
        if veh.brake_lights:
            cv2.circle(img, (int(cx), int(yr + 2)), 4, (20, 20, 255), -1)
        else:
            cv2.circle(img, (int(cx), int(yr + 2)), 2, (20, 25, 190), -1)

    def _draw_cctv_osd(self, img: np.ndarray, frame_idx: int, fps: int, scenario_name: str) -> None:
        """Renders CCTV camera timestamp, camera ID, and scenario label."""
        total_seconds = frame_idx / fps
        mins = int(total_seconds // 60)
        secs = int(total_seconds % 60)
        milli = int((total_seconds - int(total_seconds)) * 1000)

        # Semi-transparent OSD header bar
        header_bar = img[10:48, 15:FRAME_WIDTH-15].copy()
        cv2.rectangle(img, (15, 10), (FRAME_WIDTH - 15, 48), (15, 18, 20), -1)
        cv2.addWeighted(img[10:48, 15:FRAME_WIDTH-15], 0.70, header_bar, 0.30, 0, img[10:48, 15:FRAME_WIDTH-15])

        # Red recording dot
        cv2.circle(img, (32, 28), 6, (30, 30, 235), -1)
        cv2.putText(img, "REC", (44, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

        # Camera ID
        cv2.putText(img, "CAM-03 | SMART TOLL PLAZA CCTV", (95, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

        # Scenario Title
        cv2.putText(img, f"[{scenario_name.upper()}]", (470, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 210, 245), 1, cv2.LINE_AA)

        # Timestamp and Frame number
        time_str = f"TIME: {mins:02d}:{secs:02d}.{milli:03d} | FRAME: {frame_idx:05d} ({fps} FPS)"
        cv2.putText(img, time_str, (840, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_roi_overlays(self, img: np.ndarray) -> None:
        """Overlays semi-transparent colored polygons for Entry, Toll, and Exit ROIs."""
        overlay = img.copy()
        lane_rois = get_lane_rois()

        colors = {
            "entry_roi": (230, 140, 40),   # Light Blue / Cyan
            "toll_roi":  (40, 40, 240),    # Red
            "exit_roi":  (40, 220, 60),    # Green
        }

        for lane_id, rois in lane_rois.items():
            for roi_name, poly in rois.items():
                pts = np.array(poly, dtype=np.int32)
                color = colors.get(roi_name, (200, 200, 200))
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(overlay, [pts], True, color, 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)
