"""
Asset Manager for Realistic Vehicle Rendering in Smart Toll Plaza Simulation.
Manages loading, caching, scaling, rotating, and blending of transparent RGBA
vehicle sprite assets onto the CCTV toll plaza scene.
"""

import os
import glob
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class VehicleAssetManager:
    """Discovers, caches, transforms, and renders photorealistic vehicle sprite assets."""

    def __init__(self, asset_dir: str = "assets/vehicles"):
        self.asset_dir = asset_dir
        self.categories = ["car", "truck", "bus", "motorcycle"]
        # Map: category -> list of (filename, bgra_numpy_array)
        self.sprites: Dict[str, List[Tuple[str, np.ndarray]]] = {c: [] for c in self.categories}
        self.load_all_sprites()

    def load_all_sprites(self) -> None:
        """Loads all transparent PNG/WebP sprites from the asset directory."""
        for cat in self.categories:
            cat_path = os.path.join(self.asset_dir, cat)
            os.makedirs(cat_path, exist_ok=True)
            files = sorted(glob.glob(os.path.join(cat_path, "*.png")) + glob.glob(os.path.join(cat_path, "*.webp")))
            for f in files:
                img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
                    self.sprites[cat].append((os.path.basename(f), img))

        counts = {cat: len(self.sprites[cat]) for cat in self.categories}
        print(f"VehicleAssetManager loaded assets: {counts}")

    def get_sprite(self, category: str, variant_idx: int) -> Tuple[np.ndarray, str]:
        """Returns a specific or pseudo-random sprite for a given vehicle category."""
        available = self.sprites.get(category, [])
        if not available:
            # Fallback to car if requested category is missing
            available = self.sprites.get("car", [])

        if not available:
            raise RuntimeError(f"No sprite assets found in {self.asset_dir} for category '{category}'!")

        idx = variant_idx % len(available)
        name, img = available[idx]
        return img, name

    @staticmethod
    def render_vehicle_sprite(
        canvas: np.ndarray,
        sprite_bgra: np.ndarray,
        cx: float,
        cy: float,
        target_w: float,
        target_h: float,
        angle_deg: float = 0.0,
        brightness_jitter: float = 1.0
    ) -> Tuple[int, int, int, int]:
        """
        Renders a vehicle sprite with realistic perspective scaling, orientation rotation,
        natural soft contact shadow, and returns the tight bounding box [x1, y1, x2, y2].
        """
        tw = max(12, int(round(target_w)))
        th = max(18, int(round(target_h)))

        # 1. Resize sprite to target dimensions
        resized = cv2.resize(sprite_bgra, (tw, th), interpolation=cv2.INTER_AREA)

        # Apply brightness jitter if needed (slight daylight variation)
        if abs(brightness_jitter - 1.0) > 0.02:
            rgb = resized[:, :, :3].astype(np.float32) * brightness_jitter
            resized[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)

        # 2. Rotate sprite along lane perspective angle
        if abs(angle_deg) > 0.5:
            # Center of rotation
            M = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), angle_deg, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            rot_w = int((th * sin) + (tw * cos))
            rot_h = int((th * cos) + (tw * sin))
            M[0, 2] += (rot_w / 2.0) - (tw / 2.0)
            M[1, 2] += (rot_h / 2.0) - (th / 2.0)

            rotated = cv2.warpAffine(
                resized, M, (rot_w, rot_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
        else:
            rotated = resized

        rw = rotated.shape[1]
        rh = rotated.shape[0]

        # Vehicle center coordinates on canvas
        vx1 = int(round(cx - rw / 2.0))
        vy1 = int(round(cy - rh / 2.0))
        vx2 = vx1 + rw
        vy2 = vy1 + rh

        # 3. Render subtle natural contact shadow underneath vehicle
        alpha_mask = rotated[:, :, 3]
        if np.any(alpha_mask > 20):
            # Soft shadow kernel
            k_size = max(5, int(rw * 0.15) | 1)
            shadow_soft = cv2.GaussianBlur(alpha_mask, (k_size, k_size), 0)
            
            # Subtle perspective offset (simulating elevated daylight sun)
            sh_ox = int(round(3.0 * (tw / 60.0)))
            sh_oy = int(round(4.0 * (th / 100.0)))
            
            sx1 = vx1 + sh_ox
            sy1 = vy1 + sh_oy
            sx2 = sx1 + rw
            sy2 = sy1 + rh

            csx1, csy1 = max(0, sx1), max(0, sy1)
            csx2, csy2 = min(canvas.shape[1], sx2), min(canvas.shape[0], sy2)

            if csx2 > csx1 and csy2 > csy1:
                s_patch = canvas[csy1:csy2, csx1:csx2]
                sh_alpha_crop = shadow_soft[csy1 - sy1:csy2 - sy1, csx1 - sx1:csx2 - sx1]
                # Translucent contact shadow intensity (~ 35-40%)
                sh_factor = (sh_alpha_crop.astype(np.float32) / 255.0) * 0.38
                sh_factor = sh_factor[:, :, np.newaxis]
                shadow_blended = ((1.0 - sh_factor) * s_patch).astype(np.uint8)
                canvas[csy1:csy2, csx1:csx2] = shadow_blended

        # 4. Composite vehicle sprite over canvas
        cvx1, cvy1 = max(0, vx1), max(0, vy1)
        cvx2, cvy2 = min(canvas.shape[1], vx2), min(canvas.shape[0], vy2)

        if cvx2 > cvx1 and cvy2 > cvy1:
            patch = canvas[cvy1:cvy2, cvx1:cvx2]
            v_patch = rotated[cvy1 - vy1:cvy2 - vy1, cvx1 - vx1:cvx2 - vx1]
            v_alpha = (v_patch[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]
            v_bgr = v_patch[:, :, :3]
            blended = (v_alpha * v_bgr + (1.0 - v_alpha) * patch).astype(np.uint8)
            canvas[cvy1:cvy2, cvx1:cvx2] = blended

        # 5. Calculate tight visible bounding box for ground truth tracking
        # Find non-transparent pixel bounds
        ys, xs = np.where(rotated[:, :, 3] > 30)
        if len(ys) > 0:
            bbox_x1 = max(0, min(canvas.shape[1] - 1, vx1 + int(xs.min())))
            bbox_y1 = max(0, min(canvas.shape[0] - 1, vy1 + int(ys.min())))
            bbox_x2 = max(0, min(canvas.shape[1] - 1, vx1 + int(xs.max())))
            bbox_y2 = max(0, min(canvas.shape[0] - 1, vy1 + int(ys.max())))
        else:
            bbox_x1 = max(0, cvx1)
            bbox_y1 = max(0, cvy1)
            bbox_x2 = min(canvas.shape[1] - 1, cvx2)
            bbox_y2 = min(canvas.shape[0] - 1, cvy2)

        return (bbox_x1, bbox_y1, bbox_x2, bbox_y2)
