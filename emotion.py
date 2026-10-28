#!/usr/bin/env python3
"""
emotion.py - facial expression from MediaPipe FaceLandmarker blendshapes.

Uses the 52 ARKit-style blendshape coefficients to score basic expressions.
No extra model, no GPU memory: runs on CPU in ~10-20 ms.

This reads facial EXPRESSION, not inner emotional state.
"""
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

class EmotionEngine:
    def __init__(self, model_path="models/face_landmarker.task", max_faces=1):
        """Load the MediaPipe face model with blendshape output enabled."""
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)

    @staticmethod
    def _scores(b):
        """Turn 52 blendshape coefficients into six normalised expression scores."""
        smile = (b.get("mouthSmileLeft", 0) + b.get("mouthSmileRight", 0)) / 2
        frown = (b.get("mouthFrownLeft", 0) + b.get("mouthFrownRight", 0)) / 2
        brow_down = (b.get("browDownLeft", 0) + b.get("browDownRight", 0)) / 2
        brow_up = b.get("browInnerUp", 0)
        eye_wide = (b.get("eyeWideLeft", 0) + b.get("eyeWideRight", 0)) / 2
        jaw = b.get("jawOpen", 0)
        sneer = (b.get("noseSneerLeft", 0) + b.get("noseSneerRight", 0)) / 2
        press = (b.get("mouthPressLeft", 0) + b.get("mouthPressRight", 0)) / 2
        squash = b.get("mouthShrugUpper", 0)

        raw = {
            "happy": smile * 1.8,
            "surprised": eye_wide * 1.1 + jaw * 0.7,
            "angry": brow_down * 1.2 + press * 0.5,
            "sad": frown * 1.4 + brow_up * 0.8,
            "disgusted": sneer * 1.8 + squash * 0.4,
        }
        # neutral has a floor: weak signals should read as neutral, not "angry"
        peak = max(raw.values()) if raw else 0.0
        raw["neutral"] = max(0.0, 0.42 - 0.5 * peak) + 0.18
        total = sum(raw.values()) or 1.0
        return {k: round(v / total, 3) for k, v in raw.items()}

    def detect(self, frame_bgr):
        """Return the dominant expression, its scores, and the face box in pixels."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_blendshapes:
            return {"face": False, "top": "", "confidence": 0.0, "scores": {}, "box": None}

        b = {c.category_name: float(c.score) for c in res.face_blendshapes[0]}
        scores = self._scores(b)
        top = max(scores, key=scores.get)

        box = None
        if res.face_landmarks:
            pts = [(p.x * w, p.y * h) for p in res.face_landmarks[0]]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

        return {
            "face": True,
            "top": top,
            "confidence": scores[top],
            "scores": scores,
            "box": box,
        }
