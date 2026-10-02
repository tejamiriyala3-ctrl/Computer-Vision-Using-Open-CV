"""Real-time hand-gesture control: finger counter, pinch volume, air drawing.

Keys:  1 = finger counter   2 = pinch volume   3 = air drawing
       4 = face expression (happy / sad / normal)
       c = clear canvas     q / ESC = quit

Uses the MediaPipe *Tasks* API (mp.solutions was removed in mediapipe 1.x).
"""
import math
import os
import subprocess
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Landmark indices (MediaPipe's 21-point hand model).
WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20
# (tip, pip) for index, middle, ring, pinky.
FINGERS = [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
           (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]
FINGER_MCPS = [INDEX_MCP, MIDDLE_MCP, 13, PINKY_MCP]  # knuckle of each finger above
# A finger counts as raised when its tip is this many times farther from the
# wrist than its knuckle. Extended is ~1.8, curled is ~0.8, so 1.3 sits in the gap.
RAISED_RATIO = 1.3

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

MODE_NAMES = {1: "Finger Counter", 2: "Pinch Volume", 3: "Air Drawing", 4: "Face Expression"}

# Pinch ratio (thumb-index distance / hand size) that maps to 0% and 100%.
PINCH_MIN, PINCH_MAX = 0.15, 1.00
VOLUME_ALPHA = 0.25   # EMA weight for volume (lower = smoother but laggier)
DRAW_ALPHA = 0.5      # EMA weight for pen position (higher = more responsive)
CLEAR_HOLD_FRAMES = 12  # fist must be held this many frames to clear


# ---------------------------------------------------------------- setup ----
FACE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
# Expression thresholds on MediaPipe blendshape scores (0..1). Tune these if
# the on-screen scores show your face sitting on the wrong side of a line.
HAPPY_SMILE = 0.35
SAD_FROWN = 0.20
SAD_BROW = 0.35   # inner brows raised: the "worried/sad" eyebrow shape
FACE_ALPHA = 0.3


def ensure_model():
    for path, url in ((MODEL_PATH, MODEL_URL), (FACE_MODEL_PATH, FACE_MODEL_URL)):
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)} (one time only)...")
            urllib.request.urlretrieve(url, path)


def open_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        sys.exit(
            "ERROR: Could not open the webcam.\n"
            "Check macOS permissions: System Settings > Privacy & Security > Camera,\n"
            "and enable it for the app running this script (Terminal / VS Code).\n"
            "Also make sure no other app is using the camera."
        )
    return cap


def create_landmarker():
    options = vision.HandLandmarkerOptions(
        # CPU delegate: the default can try Metal/GPU, which crashed on this setup,
        # and CPU is plenty fast for a single hand.
        base_options=BaseOptions(model_asset_path=MODEL_PATH, delegate=BaseOptions.Delegate.CPU),
        # VIDEO mode tracks between frames (faster, steadier than IMAGE mode)
        # and runs synchronously, so no callback threading to reason about.
        running_mode=vision.RunningMode.VIDEO,
        # One hand: every mode assumes a single controlling hand.
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


# ------------------------------------------------------- geometry helpers ----
def dist(a, b):
    # x/y only: z from a single camera is much noisier than x/y.
    return math.hypot(a.x - b.x, a.y - b.y)


def hand_size(lm):
    # Wrist -> middle-finger knuckle. Used to normalise distances so the
    # pinch works the same whether your hand is near or far from the camera.
    return max(dist(lm[WRIST], lm[MIDDLE_MCP]), 1e-6)


def finger_states(lm):
    """Return [thumb, index, middle, ring, pinky] as booleans (True = raised).

    Fingers: tip is clearly farther from the wrist than the knuckle. The
    textbook 'tip.y < pip.y' breaks when your hand tilts; distance-to-wrist
    doesn't care about rotation. Comparing tip to PIP (my first attempt) was
    too close to call for curled fingers, so a margin against the knuckle is used.

    Thumb: it bends sideways, not up/down, so a y-comparison is meaningless,
    and an x-comparison depends on left/right hand and palm direction. Instead:
    an extended thumb tip is farther from the pinky base than its IP joint is;
    a folded thumb tip tucks across the palm toward the pinky.
    """
    wrist = lm[WRIST]
    states = [dist(lm[THUMB_TIP], lm[PINKY_MCP]) > dist(lm[THUMB_IP], lm[PINKY_MCP])]
    for (tip, _), mcp in zip(FINGERS, FINGER_MCPS):
        states.append(dist(lm[tip], wrist) > RAISED_RATIO * dist(lm[mcp], wrist))
    return states


def count_fingers(lm):
    return sum(finger_states(lm))


def get_pinch_distance(lm):
    """Thumb-tip to index-tip distance as a fraction of hand size."""
    return dist(lm[THUMB_TIP], lm[INDEX_TIP]) / hand_size(lm)


def ema(prev, new, alpha):
    """Exponential moving average: result = alpha*new + (1-alpha)*prev.

    Chosen over a plain moving average because it needs no history buffer and
    reacts faster to real movement for the same amount of smoothing.
    """
    return new if prev is None else alpha * new + (1 - alpha) * prev


# ------------------------------------------------------------ expression ----
def create_face_landmarker():
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH, delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        # Blendshapes are the model's own 0..1 scores for facial movements
        # (smile, frown, brow raise...). Much sturdier than measuring mouth
        # corner pixels ourselves, and they ignore head distance/tilt.
        output_face_blendshapes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


class ExpressionClassifier:
    """Turns blendshape scores into happy / sad / normal, with smoothing."""

    def __init__(self):
        self.scores = {"smile": None, "frown": None, "brow": None}

    def update(self, blendshapes):
        b = {c.category_name: c.score for c in blendshapes}
        raw = {
            "smile": (b.get("mouthSmileLeft", 0) + b.get("mouthSmileRight", 0)) / 2,
            "frown": (b.get("mouthFrownLeft", 0) + b.get("mouthFrownRight", 0)) / 2,
            "brow": b.get("browInnerUp", 0),
        }
        # Smooth so the label doesn't flicker between frames.
        for k, v in raw.items():
            self.scores[k] = ema(self.scores[k], v, FACE_ALPHA)
        return self.label()

    def label(self):
        sm, fr, br = self.scores["smile"], self.scores["frown"], self.scores["brow"]
        # Smile is checked first: a smile wins over a raised brow (surprise).
        if sm > HAPPY_SMILE:
            return "HAPPY"
        if fr > SAD_FROWN or (br > SAD_BROW and sm < 0.1):
            return "SAD"
        return "NORMAL"


def detect_face(face_landmarker, frame_bgr, ts):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect_for_video(image, ts)
    return result.face_blendshapes[0] if result.face_blendshapes else None


# ---------------------------------------------------------------- volume ----
class VolumeController:
    """Sets macOS volume without blocking the video loop."""

    def __init__(self):
        self.smoothed = None
        self.last_sent = None
        self.proc = None
        self.last_time = 0.0
        self.error = ""

    def update(self, pinch_ratio):
        target = float(np.interp(pinch_ratio, [PINCH_MIN, PINCH_MAX], [0, 100]))
        self.smoothed = ema(self.smoothed, target, VOLUME_ALPHA)
        level = int(round(self.smoothed))
        self._send(level)
        return level

    def _send(self, level):
        # osascript takes ~50-100 ms; running it inline would freeze the video.
        # Popen + "skip if the last call is still running" keeps FPS up, and
        # the change threshold avoids spamming the OS with 1% differences.
        now = time.time()
        if self.proc is not None and self.proc.poll() not in (None, 0):
            self.error = f"osascript failed (code {self.proc.returncode})"
        busy = self.proc is not None and self.proc.poll() is None
        changed = self.last_sent is None or abs(level - self.last_sent) >= 2
        if busy or not changed or now - self.last_time < 0.1:
            return
        self.proc = subprocess.Popen(
            ["osascript", "-e", f"set volume output volume {level}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.last_sent = level
        self.last_time = now


# --------------------------------------------------------------- drawing ----
class AirCanvas:
    def __init__(self, shape):
        self.canvas = np.zeros(shape, dtype=np.uint8)
        self.pen = None          # smoothed (x, y) pen position, or None if lifted
        self.fist_frames = 0

    def clear(self):
        self.canvas[:] = 0
        self.pen = None

    def update(self, lm, states, w, h):
        _, index, middle, ring, pinky = states
        # Thumb is ignored on purpose: it is the least reliable finger signal.
        x, y = lm[INDEX_TIP].x * w, lm[INDEX_TIP].y * h

        if not (index or middle or ring or pinky):           # fist -> clear
            self.pen = None
            self.fist_frames += 1
            if self.fist_frames == CLEAR_HOLD_FRAMES:
                # Held for a moment so a flicker in detection can't wipe your art.
                self.clear()
            return "CLEAR" if self.fist_frames >= CLEAR_HOLD_FRAMES else "FIST..."
        self.fist_frames = 0

        if index and middle and not ring and not pinky:      # two fingers -> pause
            self.pen = None  # lifting the pen avoids a stray line when you resume
            return "PAUSE"

        if index and not middle and not ring and not pinky:  # one finger -> draw
            new = (ema(self.pen[0] if self.pen else None, x, DRAW_ALPHA),
                   ema(self.pen[1] if self.pen else None, y, DRAW_ALPHA))
            if self.pen is not None:
                cv2.line(self.canvas, (int(self.pen[0]), int(self.pen[1])),
                         (int(new[0]), int(new[1])), (0, 255, 255), 6)
            self.pen = new
            return "DRAW"

        self.pen = None
        return "IDLE"

    def overlay(self, frame):
        mask = self.canvas.any(axis=2)
        frame[mask] = self.canvas[mask]


# ------------------------------------------------------------------- UI ----
def detect_hand(landmarker, frame_bgr, last_ts):
    # MediaPipe wants RGB; OpenCV gives BGR. Wrong order fails silently.
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    # Timestamps must strictly increase or MediaPipe raises an error.
    ts = max(int(time.monotonic() * 1000), last_ts + 1)
    result = landmarker.detect_for_video(image, ts)
    return (result.hand_landmarks[0] if result.hand_landmarks else None), ts


def draw_hand(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def put_text(frame, text, org, scale=0.8, color=(255, 255, 255), thick=2):
    # Dark outline first so text stays readable on any background.
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw_ui(frame, mode, fps):
    h = frame.shape[0]
    put_text(frame, f"Mode {mode}: {MODE_NAMES[mode]}", (10, 30), color=(0, 255, 255))
    put_text(frame, f"FPS: {int(fps)}", (10, 60), 0.7, (0, 255, 0))
    put_text(frame, "1/2/3/4 mode | c clear | q quit", (10, h - 15), 0.55)


def draw_volume_bar(frame, level):
    h, w = frame.shape[:2]
    x, top, bottom = w - 50, 100, h - 60
    cv2.rectangle(frame, (x, top), (x + 25, bottom), (255, 255, 255), 2)
    fill = int(bottom - (bottom - top) * level / 100)
    cv2.rectangle(frame, (x, fill), (x + 25, bottom), (0, 255, 0), -1)
    put_text(frame, f"{level}%", (x - 15, bottom + 30), 0.7)


# ------------------------------------------------------------------ main ----
def main():
    ensure_model()
    cap = open_camera()
    landmarker = create_landmarker()
    face_landmarker = create_face_landmarker()
    expression = ExpressionClassifier()
    face_ts = 0
    mode, last_ts, prev_time, fps = 1, 0, time.time(), 0.0
    volume = VolumeController()
    air = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read a frame from the camera.")
                break
            # Mirror BEFORE detection so landmarks match what you see.
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if air is None:
                air = AirCanvas(frame.shape)

            if mode == 4:
                # Face mode skips hand detection (saves CPU); the face model has
                # its own timestamp stream, which must also strictly increase.
                face_ts = max(int(time.monotonic() * 1000), face_ts + 1)
                blendshapes = detect_face(face_landmarker, frame, face_ts)
                lm = None
            else:
                lm, last_ts = detect_hand(landmarker, frame, last_ts)

            if mode == 3:
                air.overlay(frame)  # under the skeleton so the hand stays visible
            if lm:
                draw_hand(frame, lm)
                states = finger_states(lm)
                if mode == 1:
                    put_text(frame, f"Fingers: {sum(states)}", (10, 120), 1.5, (0, 255, 0), 3)
                elif mode == 2:
                    pinch = get_pinch_distance(lm)
                    level = volume.update(pinch)
                    a = (int(lm[THUMB_TIP].x * w), int(lm[THUMB_TIP].y * h))
                    b = (int(lm[INDEX_TIP].x * w), int(lm[INDEX_TIP].y * h))
                    cv2.line(frame, a, b, (255, 0, 255), 3)
                    draw_volume_bar(frame, level)
                    # Raw numbers on screen so you can see what the code sees
                    # and tune PINCH_MIN / PINCH_MAX to your hand.
                    put_text(frame, f"pinch ratio: {pinch:.2f}  (0%={PINCH_MIN} 100%={PINCH_MAX})", (10, 100), 0.6)
                    if volume.error:
                        put_text(frame, volume.error, (10, 130), 0.6, (0, 0, 255))
                elif mode == 3:
                    status = air.update(lm, states, w, h)
                    put_text(frame, status, (10, 100), 0.9, (255, 255, 0))
                    # [thumb, index, middle, ring, pinky] as 1/0 - if drawing
                    # misbehaves, this shows which finger is misread.
                    put_text(frame, "fingers: " + "".join("1" if x else "0" for x in states), (10, 130), 0.6)
                    if air.pen:
                        cv2.circle(frame, (int(air.pen[0]), int(air.pen[1])), 8, (255, 255, 255), 2)
            elif mode == 4:
                if blendshapes:
                    label = expression.update(blendshapes)
                    colors = {"HAPPY": (0, 255, 0), "SAD": (255, 120, 0), "NORMAL": (255, 255, 255)}
                    put_text(frame, label, (10, 120), 1.8, colors[label], 4)
                    sc = expression.scores
                    put_text(frame, f"smile {sc['smile']:.2f}  frown {sc['frown']:.2f}  brow {sc['brow']:.2f}",
                             (10, 150), 0.55)
                else:
                    put_text(frame, "No face detected", (10, 100), 0.7, (0, 0, 255))
            else:
                if mode == 2 and volume.smoothed is not None:
                    draw_volume_bar(frame, int(round(volume.smoothed)))
                if mode == 3:
                    air.pen = None
                    air.fist_frames = 0
                put_text(frame, "No hand detected", (10, 100), 0.7, (0, 0, 255))

            now = time.time()
            fps = ema(fps or None, 1.0 / max(now - prev_time, 1e-6), 0.1)
            prev_time = now
            draw_ui(frame, mode, fps)
            cv2.imshow("Gesture App", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                mode = key - ord("0")
            elif key == ord("c") and air is not None:
                air.clear()
    finally:
        # Always runs (even on Ctrl+C or a crash) so the camera is never left locked.
        landmarker.close()
        face_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(5):  # macOS only closes the window once events are pumped
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
