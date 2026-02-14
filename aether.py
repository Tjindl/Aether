"""
Aether — AR Whiteboard
-----------------------
Draw in the air with your index finger.
Gesture controls powered by your Handly gesture_rf.pkl model.

Drawing:
  Extend index finger only     →  Draw
  Curl index finger            →  Pause drawing

Gesture Controls:
  fist        →  Clear canvas
  rock        →  Undo last stroke
  ok          →  Save canvas as PNG
  peace       →  Cycle through colors
  thumbs_up   →  Increase brush size
  point       →  Decrease brush size
  open_palm   →  Toggle webcam feed

Keyboard:
  Q           →  Quit
  H           →  Toggle landmark overlay
  S           →  Save canvas
  C           →  Clear canvas
  Z           →  Undo
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os
from collections import deque
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH   = "gesture_rf.pkl"
SAVE_DIR     = "saved"
FONT         = cv2.FONT_HERSHEY_SIMPLEX

# Performance
CAM_W, CAM_H = 640, 480        # lower res = much faster pipeline
SMOOTH_ALPHA = 0.35             # EMA smoothing (lower = smoother but more lag)
MIN_MOVE_PX  = 3                # ignore jitter smaller than this

COLORS = [
    ("Neon Green",  (0,   255, 170)),
    ("Neon Blue",   (255, 180,   0)),
    ("Neon Pink",   (180,   0, 255)),
    ("Neon Yellow", (0,   255, 255)),
    ("Neon Red",    (80,   80, 255)),
    ("White",       (255, 255, 255)),
]

DEBOUNCE_FRAMES = 10
COOLDOWN_SEC    = 1.0

# ── Load model ────────────────────────────────────────────────────────────────

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
rf_model = bundle["model"]
le       = bundle["label_encoder"]
print(f"Gesture model loaded ✓  ({len(le.classes_)} classes)")

# ── MediaPipe ─────────────────────────────────────────────────────────────────

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=0,           # lite model — much faster on CPU
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_landmarks(landmarks, w, h):
    pts = np.array([[lm.x * w, lm.y * h, lm.z * w]
                    for lm in landmarks.landmark])
    pts -= pts[0]
    pts /= np.linalg.norm(pts[9]) + 1e-6
    return pts.flatten().reshape(1, -1)

def get_fingertip(landmarks, w, h):
    lm = landmarks.landmark[8]
    return int(lm.x * w), int(lm.y * h)

def is_drawing(landmarks):
    """Index extended + middle curled = draw mode."""
    return (landmarks.landmark[8].y < landmarks.landmark[6].y and
            landmarks.landmark[12].y > landmarks.landmark[10].y)

# ── EMA tip smoother ──────────────────────────────────────────────────────────

class TipSmoother:
    def __init__(self, alpha=SMOOTH_ALPHA):
        self.alpha = alpha
        self.sx = self.sy = None

    def update(self, x, y):
        if self.sx is None:
            self.sx, self.sy = float(x), float(y)
        else:
            self.sx = self.alpha * x + (1 - self.alpha) * self.sx
            self.sy = self.alpha * y + (1 - self.alpha) * self.sy
        return int(self.sx), int(self.sy)

    def reset(self):
        self.sx = self.sy = None

# ── Canvas ────────────────────────────────────────────────────────────────────

class Canvas:
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.layer   = np.zeros((h, w, 3), dtype=np.uint8)
        self.strokes = []
        self.current = []

    def add_point(self, pt, color, size):
        if self.current:
            lp = self.current[-1][0]
            if (pt[0]-lp[0])**2 + (pt[1]-lp[1])**2 < MIN_MOVE_PX**2:
                return                        # skip jitter
        self.current.append((pt, color, size))
        if len(self.current) >= 2:
            p1, c, s = self.current[-2]
            p2, _, _ = self.current[-1]
            cv2.line(self.layer, p1, p2, c, s, cv2.LINE_AA)

    def end_stroke(self):
        if self.current:
            self.strokes.append(self.current.copy())
            self.current = []

    def undo(self):
        if self.strokes:
            self.strokes.pop()
            self.layer[:] = 0
            for stroke in self.strokes:
                for i in range(1, len(stroke)):
                    p1, c, s = stroke[i-1]
                    p2, _, _ = stroke[i]
                    cv2.line(self.layer, p1, p2, c, s, cv2.LINE_AA)

    def clear(self):
        self.strokes = []
        self.current = []
        self.layer[:] = 0

    def save(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = os.path.join(SAVE_DIR,
               f"aether_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(path, self.layer)
        print(f"  💾 Saved → {path}")
        return path

    def composite(self, frame, show_feed):
        if show_feed:
            mask    = self.layer.astype(bool).any(axis=2)
            out     = frame.copy()
            blended = cv2.addWeighted(frame, 0.15, self.layer, 0.85, 0)
            out[mask] = blended[mask]
            return out
        return self.layer.copy()

# ── HUD ───────────────────────────────────────────────────────────────────────

def draw_hud(frame, state, h, w):
    color_name, color_val = COLORS[state["color_idx"]]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, "AETHER", (14, 34), FONT, 1.0, (0, 255, 170), 2, cv2.LINE_AA)

    # Color swatch
    cv2.circle(frame, (140, 26), 13, color_val, -1)
    cv2.circle(frame, (140, 26), 13, (255, 255, 255), 1)
    cv2.putText(frame, color_name, (158, 32), FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # Brush indicator
    bs = state["brush_size"]
    cv2.circle(frame, (290, 26), max(bs // 2, 1), (200, 200, 200), -1)
    cv2.putText(frame, f"size {bs}", (305, 32), FONT, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

    # Draw state pill
    drawing = state["drawing"]
    cv2.rectangle(frame, (w - 115, 13), (w - 10, 41),
                  (0, 200, 100) if drawing else (50, 50, 50), -1)
    cv2.putText(frame, "DRAWING" if drawing else "PAUSED",
                (w - 108, 32), FONT, 0.4,
                (0, 0, 0) if drawing else (110, 110, 110), 1, cv2.LINE_AA)

    # Bottom hint bar
    lo = frame.copy()
    cv2.rectangle(lo, (0, h - 28), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(lo, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "✌ color | 👍 size+ | ☝ size- | ✊ clear | 🤘 undo | 👌 save | ✋ cam | Q quit",
                (10, h - 8), FONT, 0.3, (80, 80, 80), 1, cv2.LINE_AA)

    # Action log
    for i, entry in enumerate(reversed(list(state["log"]))):
        col = tuple(int(c * (1.0 - i * 0.3)) for c in (0, 200, 120))
        cv2.putText(frame, entry, (w - 230, h - 60 + i * 17),
                    FONT, 0.32, col, 1, cv2.LINE_AA)

    # Flash
    if state["flash"] > 0:
        alpha = (state["flash"] / 15) * 0.28
        fo = frame.copy()
        cv2.rectangle(fo, (0, 52), (w, h - 28), (0, 255, 150), -1)
        cv2.addWeighted(fo, alpha, frame, 1 - alpha, 0, frame)
        if state["flash_msg"]:
            msg = state["flash_msg"]
            (tw, th), _ = cv2.getTextSize(msg, FONT, 0.9, 2)
            cv2.putText(frame, msg, (w//2 - tw//2, h//2 + th//2),
                        FONT, 0.9, (0, 255, 150), 2, cv2.LINE_AA)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 60)

    ret, frame = cap.read()
    if not ret:
        print("Could not open camera.")
        return

    frame  = cv2.flip(frame, 1)
    h, w   = frame.shape[:2]
    canvas = Canvas(h, w)

    state = {
        "color_idx":  0,
        "brush_size": 6,
        "drawing":    False,
        "show_feed":  True,
        "show_lm":    True,
        "flash":      0,
        "flash_msg":  "",
        "log":        deque(maxlen=3),
    }

    smoother         = TipSmoother()
    gesture_buffer   = deque(maxlen=DEBOUNCE_FRAMES)
    last_action_time = 0
    last_fired       = None
    prev_draw_state  = False

    GESTURE_ACTIONS = {
        "fist":      "clear",
        "rock":      "undo",
        "ok":        "save",
        "peace":     "color",
        "thumbs_up": "size_up",
        "point":     "size_down",
        "open_palm": "cam",
    }

    print("\n✏️  Aether — AR Whiteboard")
    print("──────────────────────────────")
    print("Extend index finger to draw, curl middle finger down to pause.")
    print("Q to quit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        draw_state = False
        gesture    = None

        if result.multi_hand_landmarks:
            lms = result.multi_hand_landmarks[0]

            if state["show_lm"]:
                mp_drawing.draw_landmarks(
                    frame, lms, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 170), thickness=1, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 180, 120), thickness=1),
                )

            # Classify gesture
            features = normalize_landmarks(lms, w, h)
            pred_idx = rf_model.predict(features)[0]
            gesture  = le.inverse_transform([pred_idx])[0]

            # Draw
            draw_state = is_drawing(lms)
            raw_tip    = get_fingertip(lms, w, h)
            tip        = smoother.update(*raw_tip)
            color_val  = COLORS[state["color_idx"]][1]

            cv2.circle(frame, tip, state["brush_size"] // 2 + 3, color_val, 2, cv2.LINE_AA)
            cv2.circle(frame, tip, 2, (255, 255, 255), -1)

            if draw_state:
                if not prev_draw_state:
                    smoother.reset()
                    smoother.update(*raw_tip)
                canvas.add_point(tip, color_val, state["brush_size"])
            else:
                if prev_draw_state:
                    canvas.end_stroke()
                smoother.reset()

            # Gesture debounce
            if gesture in GESTURE_ACTIONS:
                gesture_buffer.append(gesture)
            else:
                gesture_buffer.clear()

            if (len(gesture_buffer) == DEBOUNCE_FRAMES and
                    len(set(gesture_buffer)) == 1):
                now    = time.time()
                stable = gesture_buffer[0]

                if now - last_action_time > COOLDOWN_SEC or stable != last_fired:
                    action = GESTURE_ACTIONS[stable]
                    msg    = ""

                    if action == "clear":
                        canvas.clear();  msg = "Canvas Cleared"
                    elif action == "undo":
                        canvas.undo();   msg = "Undo ↩"
                    elif action == "save":
                        canvas.save();   msg = "Saved 💾"
                    elif action == "color":
                        state["color_idx"] = (state["color_idx"] + 1) % len(COLORS)
                        msg = f"Color: {COLORS[state['color_idx']][0]}"
                    elif action == "size_up":
                        state["brush_size"] = min(state["brush_size"] + 2, 30)
                        msg = f"Brush: {state['brush_size']}"
                    elif action == "size_down":
                        state["brush_size"] = max(state["brush_size"] - 2, 2)
                        msg = f"Brush: {state['brush_size']}"
                    elif action == "cam":
                        state["show_feed"] = not state["show_feed"]
                        msg = f"Cam: {'ON' if state['show_feed'] else 'OFF'}"

                    if msg:
                        state["log"].append(f"{time.strftime('%H:%M:%S')}  {msg}")
                        state["flash"]     = 15
                        state["flash_msg"] = msg
                        last_action_time   = now
                        last_fired         = stable
                        print(f"  ▶ {msg}")

                gesture_buffer.clear()

        else:
            if prev_draw_state:
                canvas.end_stroke()
            smoother.reset()
            gesture_buffer.clear()

        prev_draw_state  = draw_state
        state["drawing"] = draw_state

        if state["flash"] > 0:
            state["flash"] -= 1

        output = canvas.composite(frame, state["show_feed"])
        draw_hud(output, state, h, w)
        cv2.imshow("Aether", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            canvas.end_stroke(); break
        elif key == ord("h"):
            state["show_lm"] = not state["show_lm"]
        elif key == ord("c"):
            canvas.clear(); state["log"].append("Canvas cleared")
        elif key == ord("z"):
            canvas.undo();  state["log"].append("Undo")
        elif key == ord("s"):
            canvas.save();  state["log"].append("Saved")

    cap.release()
    cv2.destroyAllWindows()
    print("\nAether closed.")


if __name__ == "__main__":
    main()