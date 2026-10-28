#!/usr/bin/env python3
"""
depth.py - metric monocular depth for distance measurement and point clouds.

Depth Anything V2 Metric Indoor (Small) predicts depth in metres directly, so no
calibration or reference object is needed. Indoor scenes only - it is trained on
Hypersim/Virtual KITTI interiors.
"""
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


class DepthEngine:
    def __init__(self, model_id=MODEL_ID, device=0):
        """Load the metric depth model onto the GPU."""
        self.model_id = model_id
        self.device = device
        self.pipe = pipeline("depth-estimation", model=model_id, device=device)

    def estimate(self, frame_bgr):
        """Returns a float32 depth map in metres, same size as the frame."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = self.pipe(Image.fromarray(rgb))
        d = out["predicted_depth"]
        if isinstance(d, torch.Tensor):
            d = d.detach().float().cpu().numpy()
        d = np.squeeze(d)
        h, w = frame_bgr.shape[:2]
        if d.shape[:2] != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        return d.astype(np.float32)

    @staticmethod
    def distance_at(depth, x, y, radius=6):
        """Median depth in a small window, in metres. Robust to single-pixel noise."""
        h, w = depth.shape[:2]
        x = max(0, min(w - 1, int(x)))
        y = max(0, min(h - 1, int(y)))
        patch = depth[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
        return float(np.median(patch))

    @staticmethod
    def colourise(depth, lo=0.5, hi=8.0):
        """Turbo heatmap of metric depth, same size as the input map."""
        norm = np.clip((depth - lo) / max(1e-6, (hi - lo)), 0, 1)
        img = (norm * 255).astype(np.uint8)
        return cv2.applyColorMap(img, cv2.COLORMAP_TURBO)

    @staticmethod
    def pointcloud(depth, frame_bgr, rows, cols, hfov_deg=60.0):
        """Back-project a depth grid to 3D points with colour. Returns float32 (N,6)."""
        h, w = depth.shape[:2]
        ys = np.linspace(0, h - 1, rows).astype(int)
        xs = np.linspace(0, w - 1, cols).astype(int)
        fx = w / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
        fy = fx
        cx, cy = w / 2.0, h / 2.0
        pts = []
        small = cv2.resize(frame_bgr, (cols, rows), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for r, yy in enumerate(ys):
            for c, xx in enumerate(xs):
                z = depth[yy, xx]
                if not np.isfinite(z) or z <= 0.05 or z > 25:
                    continue
                X = (xx - cx) * z / fx
                Y = (yy - cy) * z / fy
                pts.append((X, -Y, z,
                            rgb[r, c, 0], rgb[r, c, 1], rgb[r, c, 2]))
        if not pts:
            return np.zeros((0, 6), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32)
