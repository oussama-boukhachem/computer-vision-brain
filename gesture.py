#!/usr/bin/env python3
"""
gesture.py - hand landmarks, finger states and held-gesture detection.

Uses MediaPipe Tasks HandLandmarker (21 landmarks per hand). Finger states are
derived from joint-chain straightness, which is scale and rotation invariant.

    from gesture import HandTracker
    tracker = HandTracker(hold=1.2)
    hands = tracker.detect(frame_bgr)   # list of dicts, see HandTracker.detect
"""
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

CHAINS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

# straightness above this = finger extended, below = curled
EXTENDED = 0.90
CURLED = 0.78

ACCENT = (255, 214, 0)      # BGR cyan-ish
FIRED = (120, 255, 120)


def _straightness(pts, chain):
    """Ratio between the straight chord and the sum of joint segments (0..1)."""
    total = 0.0
    for i in range(len(chain) - 1):
        a = pts[chain[i]]
        b = pts[chain[i + 1]]
        total += ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    if total <= 1e-6:
        return 0.0
    p0 = pts[chain[0]]
    p1 = pts[chain[-1]]
    chord = ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5
    return chord / total


def _name_from_fingers(ext):
    """Map extended-finger set to a gesture name."""
    key = tuple(sorted(k for k, v in ext.items() if v))
    table = {
        (): "fist",
        ("thumb",): "thumbs_up",
        ("index",): "point",
        ("index", "middle"): "peace",
        ("index", "middle", "ring"): "three",
        ("index", "middle", "ring", "pinky"): "four",
        ("index", "pinky"): "rock",
        ("pinky", "thumb"): "shaka",
        ("index", "middle", "pinky", "ring", "thumb"): "palm",
    }
    return table.get(key, "unknown")


class HeldGesture:
    """Fires once when the same gesture is held continuously for a duration."""

    def __init__(self, hold=1.2):
        """Create a hold tracker; hold is the seconds a gesture must persist."""
        self.hold = max(0.05, float(hold))
        self.current = None
        self.since = 0.0
        self.progress = 0.0
        self._fired_for = None

    def update(self, gesture, now):
        """Returns (fired_gesture_or_None, progress_0_to_1)."""
        if not gesture:
            self.current = None
            self.since = 0.0
            self.progress = 0.0
            self._fired_for = None
            return None, 0.0
        if gesture != self.current:
            self.current = gesture
            self.since = now
            self._fired_for = None
        elapsed = now - self.since
        self.progress = min(1.0, elapsed / self.hold)
        fired = None
        if elapsed >= self.hold and self._fired_for != gesture:
            fired = gesture
            self._fired_for = gesture
        return fired, self.progress


class HandTracker:
    def __init__(self, model_path="models/hand_landmarker.task", max_hands=2,
                 min_conf=0.5, hold=1.2):
        """Load the MediaPipe hand model and set the hold duration."""
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self.hold = hold
        self._held = {}

    def reset(self):
        """Forget per-hand hold state (call when the camera stops)."""
        self._held.clear()

    def detect(self, frame_bgr, now=None):
        """Returns a list of hands: handedness, px landmarks, fingers, gesture, progress."""
        now = time.time() if now is None else now
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect(mp_img)

        hands = []
        if not res.hand_landmarks:
            for state in self._held.values():
                state.update(None, now)
            return hands

        for i, lms in enumerate(res.hand_landmarks):
            pts = [(lm.x * w, lm.y * h) for lm in lms]
            ext = {}
            for name, chain in CHAINS.items():
                ratio = _straightness(pts, chain)
                ext[name] = ratio > (0.86 if name == "thumb" else EXTENDED)
            # thumb tip to index tip gap, normalized by hand size (for "ok")
            scale = max(1.0, ((pts[0][0] - pts[9][0]) ** 2 + (pts[0][1] - pts[9][1]) ** 2) ** 0.5)
            pinch = ((pts[4][0] - pts[8][0]) ** 2 + (pts[4][1] - pts[8][1]) ** 2) ** 0.5 / scale
            gesture = _name_from_fingers(ext)
            if pinch < 0.45 and ext["middle"] and ext["ring"] and ext["pinky"]:
                gesture = "ok"
                ext = dict(ext)
                ext["thumb"] = True
                ext["index"] = True

            label = "hand_%d" % i
            state = self._held.get(label)
            if state is None:
                state = HeldGesture(self.hold)
                self._held[label] = state
            fired, progress = state.update(gesture, now)

            hands.append({
                "slot": label,
                "handedness": (res.handedness[i][0].category_name
                               if res.handedness and i < len(res.handedness) else "?"),
                "landmarks": pts,
                "fingers": ext,
                "gesture": gesture,
                "progress": round(progress, 2),
                "fired": fired,
            })
        return hands


def draw_hands(frame, hands):
    """Draw skeleton, finger dots and the held-gesture progress bar in place."""
    for hand in hands:
        pts = hand["landmarks"]
        ready = hand["progress"] >= 1.0
        colour = FIRED if ready else ACCENT
        for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS:
            a = pts[conn.start]
            b = pts[conn.end]
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), colour, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, colour, -1, cv2.LINE_AA)

        wx, wy = int(pts[0][0]), int(pts[0][1])
        label = "%s  %s" % (hand["gesture"].replace("_", " "), hand["handedness"])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (wx - 6, wy + 12), (wx + tw + 10, wy + th + 26), (12, 16, 22), -1)
        cv2.putText(frame, label, (wx, wy + th + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        # hold progress bar
        bar_w = max(60, tw + 16)
        filled = int(bar_w * hand["progress"])
        cv2.rectangle(frame, (wx - 6, wy + th + 30), (wx - 6 + bar_w, wy + th + 36), (40, 48, 58), -1)
        if filled > 0:
            cv2.rectangle(frame, (wx - 6, wy + th + 30), (wx - 6 + filled, wy + th + 36), colour, -1)
