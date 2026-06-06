"""
main.py — Real-Time Multimodal Driver Drowsiness Detection
==========================================================
Architecture : MediaPipe Face Mesh + CNN (MobileNetV2) + Sensor Fusion
Tech stack   : Python 3.12 | TensorFlow 2.18 | MediaPipe 0.10 | OpenCV 4.13
                Pygame 2.6 | NumPy | SciPy

Usage
-----
  python main.py                                    # integrated webcam (index 0)
  python main.py --source 0                         # same, explicit
  python main.py --source 1                         # external webcam / DroidCam
  python main.py --source phone                     # IP Webcam at default URL
  python main.py --source url http://IP:PORT/video  # custom IP-camera URL

Improvements over v1
---------------------
[FIX-1]  Alarm loops until danger is gone (was playing only once)
[FIX-2]  alert_sent reset only on stable NON-DANGER (not every frame)
[FIX-3]  CNN runs every N frames — big speed boost
[FIX-4]  alarm.wav auto-generated at startup if missing
[FIX-5]  Proper alarm stop when status drops below DANGER
[NEW-1]  30-second adaptive calibration (personal EAR / MAR baseline)
[NEW-2]  Yawn counter & prolonged eye-closure counter displayed on screen
[NEW-3]  Landmark drawing (eyes + mouth outline) for visual feedback
[NEW-4]  FPS display
[NEW-5]  Graceful "No face detected" keeps last values on screen
[NEW-6]  Progress-bar colour smoothly follows fatigue level
[NEW-7]  Separate eye-region crop fed to CNN eye model (better accuracy)
[NEW-8]  --source CLI argument: webcam index, 'phone', or 'url <URL>'
[NEW-9]  Normalised 640×480 display — consistent UI on any camera source
[NEW-10] IP-camera buffer flush — eliminates accumulated lag
[NEW-11] Adaptive SKIP_FRAMES — 5 for webcam, 8 for IP camera
"""

import argparse
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pygame
import time
import os
from scipy.spatial import distance
from scipy.io import wavfile


# ============================================================
# CLI ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser(description="Drowsiness Detection System")
parser.add_argument(
    "--source", nargs="+", default=["0"],
    help="Camera source: '0' (webcam), '1', 'phone', or 'url http://IP:PORT/video'"
)
args = parser.parse_args()

# Resolve source → (camera_source, is_ip_camera)
_src = args.source
if len(_src) == 1 and _src[0] == "phone":
    # Default IP Webcam URL — edit to match your phone's IP
    CAMERA_SOURCE  = "http://192.168.0.166:8080/video"
    IS_IP_CAMERA   = True
elif len(_src) == 2 and _src[0] == "url":
    CAMERA_SOURCE  = _src[1]
    IS_IP_CAMERA   = True
else:
    # Numeric index (integrated or USB webcam)
    try:
        CAMERA_SOURCE = int(_src[0])
    except ValueError:
        CAMERA_SOURCE = _src[0]
    IS_IP_CAMERA = False

print(f"📷  Camera source : {CAMERA_SOURCE}  |  IP-camera mode : {IS_IP_CAMERA}")

# ============================================================
# CONSTANTS & TUNEABLE PARAMETERS
# ============================================================
# Display resolution — ALL frames are resized to this before any processing.
# Guarantees identical UI layout regardless of camera source.
DISPLAY_W = 640
DISPLAY_H = 480

# CNN skip-frame rate: fewer predictions = more CPU/GPU headroom
# IP cameras are slower overall, so we skip more aggressively.
SKIP_FRAMES         = 8 if IS_IP_CAMERA else 5

# How many frames to discard from the IP-cam buffer each cycle.
# MJPEG streams accumulate buffered frames → causes visible lag.
IP_FLUSH_FRAMES     = 3

CALIB_SECONDS       = 30      # Adaptive calibration window
EAR_THRESHOLD_DEF   = 0.25    # Default EAR (overridden after calib.)
MAR_THRESHOLD_DEF   = 0.60    # Default MAR (overridden after calib.)
EYE_CLOSE_FRAMES    = 15      # Consecutive frames below EAR → "closure event"
YAWN_FRAMES         = 20      # Consecutive frames above MAR → "yawn event"
ALARM_WAV           = "alarm.wav"

# Fusion weights (must sum to 1.0)
W_EYE   = 0.50
W_MOUTH = 0.30
W_HEAD  = 0.20

# CNN weights inside each score (geometric vs. CNN)
# Eye: trust CNN heavily (0.70) — it was trained on 50k images and is robust
# to bad cameras and glasses reflections. Raw EAR geometry gets only 0.30.
# Mouth: keep 50/50 since the mouth CNN dataset is smaller (1.2k images).
W_CNN_EYE   = 0.70   # CNN contribution to eye score
W_GEO_EYE   = 0.30   # EAR geometry contribution to eye score
W_CNN_MOUTH = 0.50   # CNN contribution to mouth score
W_GEO_MOUTH = 0.50   # MAR geometry contribution to mouth score

# EAR smoothing — exponential moving average applied to raw EAR each frame.
# Prevents a single blurry frame from spiking the fatigue score.
EAR_SMOOTH_ALPHA = 0.4   # lower = smoother but slower to react

# ============================================================
# LANDMARK INDICES (MediaPipe 478-point model)
# ============================================================
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]

# Indices for drawing eye contours (full ring)
LEFT_EYE_DRAW  = [362, 382, 381, 380, 374, 373, 390, 249,
                  263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_DRAW = [33, 7, 163, 144, 145, 153, 154, 155,
                  133, 173, 157, 158, 159, 160, 161, 246]
MOUTH_DRAW     = [61, 185, 40, 39, 37, 0, 267, 269,
                  270, 409, 291, 375, 321, 405, 314, 17,
                  84, 181, 91, 146]


# ============================================================
# ALARM GENERATION
# ============================================================
def _generate_alarm_wav(path: str, duration: float = 3.0, sr: int = 44100) -> None:
    """Create a pulsed two-tone siren and save it to path."""
    pulse = int(sr * 0.25)
    fade  = int(sr * 0.05)
    n_samples = int(sr * duration)
    signal = np.zeros(n_samples, dtype=np.float32)
    for i, start in enumerate(range(0, n_samples, pulse)):
        end  = min(start + pulse, n_samples)
        seg  = np.arange(end - start)
        freq = 880 if i % 2 == 0 else 440
        wave = np.sin(2 * np.pi * freq * seg / sr).astype(np.float32)
        env  = np.ones(len(wave), dtype=np.float32)
        f    = min(fade, len(wave) // 2)
        env[:f]  = np.linspace(0, 1, f)
        env[-f:] = np.linspace(1, 0, f)
        signal[start:end] = wave * env
    peak = np.max(np.abs(signal)) or 1.0
    pcm = (signal / peak * 0.9 * 32767).astype(np.int16)
    wavfile.write(path, sr, pcm)
    print(f"✅  alarm.wav generated at {os.path.abspath(path)}")


# ============================================================
# INITIALISATION
# ============================================================
print("⏳  Initialising pygame mixer …")
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

# Auto-generate alarm.wav if absent
if not os.path.isfile(ALARM_WAV):
    print("⚠️   alarm.wav not found — generating …")
    _generate_alarm_wav(ALARM_WAV)

# Pre-load alarm sound
try:
    alarm_sound = pygame.mixer.Sound(ALARM_WAV)
    alarm_channel = None          # will hold the Channel object when playing
    print("✅  Alarm sound loaded.")
except Exception as e:
    alarm_sound = None
    alarm_channel = None
    print(f"⚠️   Could not load alarm: {e}")

print("⏳  Loading CNN models …")
try:
    eye_model   = tf.keras.models.load_model('models/eye_model.keras')
    mouth_model = tf.keras.models.load_model('models/mouth_model.keras')
    print("✅  Models loaded.")
except Exception as e:
    print(f"❌  Model loading failed: {e}")
    raise SystemExit(1)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

os.makedirs('outputs', exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def eye_aspect_ratio(landmarks, eye_indices, w, h) -> float:
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def mouth_aspect_ratio(landmarks, mouth_indices, w, h) -> float:
    """
    MAR uses:
      pts[0]=left corner (61), pts[1]=right corner (291)
      pts[2]=top-inner (39),   pts[6]=bottom-inner (269)
      pts[3]=top-mid  (181),   pts[7]=bottom-mid   (405)
    Vertical distances / horizontal width.
    """
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in mouth_indices]
    A = distance.euclidean(pts[2], pts[6])   # inner vertical
    B = distance.euclidean(pts[3], pts[7])   # mid vertical
    C = distance.euclidean(pts[0], pts[1])   # horizontal width
    return (A + B) / (2.0 * C + 1e-6)


def get_head_tilt(landmarks, w, h):
    """Returns (roll_deg, pitch_normalised)."""
    left_eye  = (landmarks[133].x * w, landmarks[133].y * h)
    right_eye = (landmarks[362].x * w, landmarks[362].y * h)
    nose      = (landmarks[1].x   * w, landmarks[1].y   * h)
    chin      = (landmarks[152].x * w, landmarks[152].y * h)
    dx   = right_eye[0] - left_eye[0]
    dy   = right_eye[1] - left_eye[1]
    roll  = abs(np.degrees(np.arctan2(dy, dx)))
    pitch = abs((chin[1] - nose[1]) / (h * 0.1 + 1e-6))
    return roll, pitch


def preprocess_frame(frame) -> np.ndarray:
    """Full-frame preprocessing for CNN (fallback / mouth model)."""
    img = cv2.resize(frame, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype(np.float32) / 255.0, axis=0)


def preprocess_eye_region(frame, landmarks, w, h) -> np.ndarray:
    """
    Crop a bounding box around both eyes and resize to 64×64.
    Feeding the CNN a focused region improves accuracy significantly
    compared to using the full frame.
    """
    all_eye_idx = LEFT_EYE + RIGHT_EYE
    xs = [int(landmarks[i].x * w) for i in all_eye_idx]
    ys = [int(landmarks[i].y * h) for i in all_eye_idx]
    pad = 20
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w, max(xs) + pad)
    y2 = min(h, max(ys) + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return preprocess_frame(frame)
    img = cv2.resize(crop, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype(np.float32) / 255.0, axis=0)


def preprocess_face_region(frame, landmarks, w, h) -> np.ndarray:
    """
    Crop a tight bounding box around the full face and resize to 64×64.

    The mouth (yawn) CNN was trained on full-face images from the Yawn Eye
    Dataset New — it expects the whole face, not just the mouth region.
    Feeding a mouth crop would destroy the spatial context the model relies on.
    """
    # Use a broad set of face-boundary landmarks for a reliable bounding box
    face_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                323, 361, 288, 397, 365, 379, 378, 400, 377,
                152, 148, 176, 149, 150, 136, 172, 58, 132,
                93,  234, 127,  162,  21,  54, 103,  67,  109]
    xs = [int(landmarks[i].x * w) for i in face_idx]
    ys = [int(landmarks[i].y * h) for i in face_idx]
    pad = 10
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w, max(xs) + pad)
    y2 = min(h, max(ys) + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return preprocess_frame(frame)
    img = cv2.resize(crop, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype(np.float32) / 255.0, axis=0)


def save_screenshot(frame, level: str) -> None:
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = f'outputs/alert_{level}_{ts}.jpg'
    cv2.imwrite(path, frame)
    print(f"📸  Screenshot saved → {path}")


def start_alarm() -> None:
    global alarm_channel
    if alarm_sound and (alarm_channel is None or not alarm_channel.get_busy()):
        alarm_channel = alarm_sound.play(loops=-1)   # loop until stopped


def stop_alarm() -> None:
    global alarm_channel
    if alarm_channel is not None:
        alarm_channel.stop()
        alarm_channel = None


def draw_landmarks_outline(frame, landmarks, indices, w, h, color, thickness=1):
    """Draw a closed polygon through a set of landmark indices."""
    pts = np.array(
        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices],
        dtype=np.int32
    )
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)


def interpolate_color(pct: float):
    """
    Map 0→100 fatigue % to a smooth BGR colour:
      0-30  : green  (0,200,0)
      30-60 : orange (0,165,255)
      60-100: red    (0,0,255)
    """
    if pct < 30:
        r = int(pct / 30 * 200)
        return (0, 200 - r, r)           # green → approaching orange
    elif pct < 60:
        t = (pct - 30) / 30
        g = int((1 - t) * 165)
        b = int(t * 255)
        return (0, g, 90 + b // 2)
    else:
        return (0, 0, 255)               # full red


# ============================================================
# STATE VARIABLES
# ============================================================
# Calibration
calib_ear_samples: list = []
calib_mar_samples: list = []
calibrated        = False
calib_start       = time.time()
EAR_THRESHOLD     = EAR_THRESHOLD_DEF
MAR_THRESHOLD     = MAR_THRESHOLD_DEF

# Alert state
alert_sent        = False
danger_frame_count = 0
DANGER_CONFIRM    = 10      # frames before triggering alarm (debounce)

# Counters
yawn_count          = 0
eye_closure_count   = 0
consec_eye_close    = 0    # consecutive frames with eye closed
consec_mouth_open   = 0    # consecutive frames with mouth open
yawn_active         = False
eye_close_active    = False

# CNN cache (refreshed every SKIP_FRAMES)
frame_counter = 0
eye_pred      = 0.0
mouth_pred    = 0.0

# Smoothing (exponential moving average on fatigue score)
fatigue_smooth = 0.0
SMOOTH_ALPHA   = 0.3

# EAR smoothing — separate EMA to de-noise raw geometry before fusion
ear_smooth     = EAR_THRESHOLD_DEF   # initialise at sane value

# Previous metric values (shown when face temporarily lost)
last_ear   = 0.0
last_mar   = 0.0
last_roll  = 0.0
last_pitch = 0.0
last_fatigue_pct = 0

# FPS
fps_time   = time.time()
fps_val    = 0.0
fps_count  = 0

start_time = time.time()

# ============================================================
# CAMERA
# ============================================================
cap = cv2.VideoCapture(CAMERA_SOURCE)

if IS_IP_CAMERA:
    # For IP cameras: reduce OpenCV's internal buffer to 1 frame so we
    # always get the most recent frame instead of stale buffered ones.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Request a moderate resolution from the phone stream — less data
    # to decode over WiFi means lower latency.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
else:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print(f"❌  Cannot open camera: {CAMERA_SOURCE}")
    print("    Webcam: try --source 1  |  Phone: check IP address in URL")
    raise SystemExit(1)

print("🎥  System started — Press 'q' to quit")

# Pre-create the window at a fixed size so all UI elements are always visible.
# cv2.WINDOW_NORMAL allows user resizing; resizeWindow sets the initial size.
_win_title = f"Drowsiness Detection System  |  {'IP Camera' if IS_IP_CAMERA else f'Webcam [{CAMERA_SOURCE}]'}"
cv2.namedWindow(_win_title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(_win_title, DISPLAY_W, DISPLAY_H)


# ============================================================
# MAIN LOOP
# ============================================================
while True:
    # ---- IP-camera buffer flush ----
    # Discard stale buffered frames so we always process the LATEST frame.
    # Without this, lag accumulates and the display falls further and further
    # behind real time.  For local webcams this is a no-op (grab() is instant).
    if IS_IP_CAMERA:
        for _ in range(IP_FLUSH_FRAMES):
            cap.grab()   # fast decode-skip (no image copy)

    ret, frame = cap.read()
    if not ret:
        print("⚠️   Frame capture failed — retrying …")
        time.sleep(0.05)
        continue

    # ---- Normalise resolution ----
    # Always work on DISPLAY_W × DISPLAY_H internally.
    # This guarantees identical landmark positions and UI layout
    # regardless of what resolution the camera actually delivers.
    if frame.shape[1] != DISPLAY_W or frame.shape[0] != DISPLAY_H:
        frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H),
                           interpolation=cv2.INTER_LINEAR)

    h, w  = DISPLAY_H, DISPLAY_W          # always 480, 640
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---- FPS calculation ----
    fps_count += 1
    now = time.time()
    if now - fps_time >= 1.0:
        fps_val   = fps_count / (now - fps_time)
        fps_count = 0
        fps_time  = now

    # ---- Face Mesh ----
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # ---- Geometric metrics ----
        ear = (eye_aspect_ratio(landmarks, LEFT_EYE,  w, h) +
               eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)) / 2.0
        mar        = mouth_aspect_ratio(landmarks, MOUTH, w, h)
        roll, pitch = get_head_tilt(landmarks, w, h)

        # Cache for "face lost" fallback
        last_ear, last_mar, last_roll, last_pitch = ear, mar, roll, pitch

        # ---- Adaptive calibration ----
        elapsed_calib = now - calib_start
        if not calibrated:
            calib_ear_samples.append(ear)
            calib_mar_samples.append(mar)
            if elapsed_calib >= CALIB_SECONDS:
                # --- EAR threshold ---
                # Use the 15th percentile of baseline EAR samples.
                # Percentile is far more robust than mean-std: a few blink frames
                # or noisy frames from a bad webcam don't skew the result.
                # 15th pct ≈ "the EAR that 85% of your open-eye frames are above"
                EAR_THRESHOLD = float(np.percentile(calib_ear_samples, 15))
                EAR_THRESHOLD = max(EAR_THRESHOLD, 0.15)   # hard floor (safety)
                # --- MAR threshold ---
                # 85th percentile of resting mouth → triggered only by clear yawns
                MAR_THRESHOLD = float(np.percentile(calib_mar_samples, 85))
                MAR_THRESHOLD = min(MAR_THRESHOLD, 0.90)   # hard ceiling
                calibrated = True
                print(f"✅  Calibration done — EAR_thresh={EAR_THRESHOLD:.3f} "
                      f"(15th pct)  MAR_thresh={MAR_THRESHOLD:.3f} (85th pct)")

        # ---- CNN prediction (every SKIP_FRAMES) ----
        # eye_model  → trained on MRL Eye Dataset (cropped eye images) → eye crop
        # mouth_model → trained on Yawn Eye Dataset (full-face images) → face crop
        frame_counter += 1
        if frame_counter % SKIP_FRAMES == 0:
            eye_input   = preprocess_eye_region(frame, landmarks, w, h)
            mouth_input = preprocess_face_region(frame, landmarks, w, h)
            eye_pred    = float(eye_model.predict(eye_input,   verbose=0)[0][0])
            mouth_pred  = float(mouth_model.predict(mouth_input, verbose=0)[0][0])

        # ---- Fusion score ----
        # Smooth raw EAR before using it — removes single-frame spikes from
        # glasses reflections, blinks, or poor camera quality.
        ear_smooth  = EAR_SMOOTH_ALPHA * ear + (1 - EAR_SMOOTH_ALPHA) * ear_smooth

        ear_score   = max(0.0, (EAR_THRESHOLD - ear_smooth) / (EAR_THRESHOLD + 1e-6))
        mar_score   = max(0.0, (mar - MAR_THRESHOLD)        / (MAR_THRESHOLD + 1e-6))
        eye_final   = W_CNN_EYE   * eye_pred   + W_GEO_EYE   * ear_score
        mouth_final = W_CNN_MOUTH * mouth_pred + W_GEO_MOUTH * mar_score
        head_final  = min(1.0, roll / 30.0 + pitch / 5.0)

        fatigue_raw    = W_EYE * eye_final + W_MOUTH * mouth_final + W_HEAD * head_final
        fatigue_smooth = SMOOTH_ALPHA * fatigue_raw + (1 - SMOOTH_ALPHA) * fatigue_smooth
        fatigue_pct    = int(fatigue_smooth * 100)
        last_fatigue_pct = fatigue_pct

        # ---- Event counters ----
        # Use the smoothed EAR for event counting too (same de-noising benefit)
        # Eye closure events
        if ear_smooth < EAR_THRESHOLD:
            consec_eye_close += 1
        else:
            if eye_close_active and consec_eye_close >= EYE_CLOSE_FRAMES:
                eye_closure_count += 1
            eye_close_active  = False
            consec_eye_close  = 0

        if consec_eye_close >= EYE_CLOSE_FRAMES:
            eye_close_active = True

        # Yawn events
        if mar > MAR_THRESHOLD:
            consec_mouth_open += 1
        else:
            if yawn_active and consec_mouth_open >= YAWN_FRAMES:
                yawn_count += 1
            yawn_active       = False
            consec_mouth_open = 0

        if consec_mouth_open >= YAWN_FRAMES:
            yawn_active = True

        # ---- Alert level ----
        if fatigue_pct < 30:
            status = "NORMAL"
        elif fatigue_pct < 60:
            status = "WARNING"
        else:
            status = "DANGER"

        bar_color = interpolate_color(fatigue_pct)

        # ---- Alarm logic (debounced) ----
        if status == "DANGER":
            danger_frame_count += 1
            if danger_frame_count >= DANGER_CONFIRM and not alert_sent:
                start_alarm()
                save_screenshot(frame, "DANGER")
                alert_sent = True
        else:
            danger_frame_count = 0
            if alert_sent:
                stop_alarm()
                alert_sent = False

        # ---- Landmark overlay ----
        lm_color = (0, 255, 0) if status == "NORMAL" else \
                   (0, 165, 255) if status == "WARNING" else (0, 80, 255)
        draw_landmarks_outline(frame, landmarks, LEFT_EYE_DRAW,  w, h, lm_color)
        draw_landmarks_outline(frame, landmarks, RIGHT_EYE_DRAW, w, h, lm_color)
        draw_landmarks_outline(frame, landmarks, MOUTH_DRAW,     w, h, lm_color)

        # ============================================================
        # UI RENDERING
        # ============================================================

        # --- Top panel background ---
        cv2.rectangle(frame, (0, 0), (w, 100), (18, 18, 18), -1)

        # --- Status text ---
        status_colors = {"NORMAL": (50, 220, 50), "WARNING": (0, 165, 255), "DANGER": (50, 50, 255)}
        cv2.putText(frame, f"STATUS: {status}", (12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_colors[status], 2, cv2.LINE_AA)

        # --- Fatigue % ---
        cv2.putText(frame, f"Fatigue: {fatigue_pct}%", (12, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2, cv2.LINE_AA)

        # --- Progress bar ---
        bar_w = int((w - 20) * min(fatigue_smooth, 1.0))
        cv2.rectangle(frame, (10, 78), (w - 10, 92), (50, 50, 50), -1)
        if bar_w > 0:
            cv2.rectangle(frame, (10, 78), (10 + bar_w, 92), bar_color, -1)

        # --- Timer (top-right) ---
        elapsed    = int(now - start_time)
        mins, secs = divmod(elapsed, 60)
        cv2.putText(frame, f"{mins:02d}:{secs:02d}", (w - 85, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2, cv2.LINE_AA)

        # --- FPS (top-right, below timer) ---
        cv2.putText(frame, f"FPS:{fps_val:.0f}", (w - 80, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        # --- Counters (mid-right panel) ---
        counter_panel_x = w - 160
        cv2.rectangle(frame, (counter_panel_x - 5, 100), (w, 160), (25, 25, 25), -1)
        cv2.putText(frame, f"Yawns     : {yawn_count}", (counter_panel_x, 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Eye closes: {eye_closure_count}", (counter_panel_x, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # --- Calibration overlay ---
        if not calibrated:
            calib_pct  = min(100, int(elapsed_calib / CALIB_SECONDS * 100))
            remain_sec = max(0, int(CALIB_SECONDS - elapsed_calib))
            overlay    = frame.copy()
            cv2.rectangle(overlay, (0, h // 2 - 30), (w, h // 2 + 30), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"CALIBRATING ... {remain_sec}s remaining ({calib_pct}%)",
                        (w // 2 - 220, h // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2, cv2.LINE_AA)
            # Draw calibration progress bar
            cal_bar = int((w - 20) * calib_pct / 100)
            cv2.rectangle(frame, (10, h // 2 + 38), (w - 10, h // 2 + 48), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, h // 2 + 38), (10 + cal_bar, h // 2 + 48), (50, 220, 255), -1)

        # --- Bottom metrics strip ---
        cv2.rectangle(frame, (0, h - 28), (w, h), (18, 18, 18), -1)
        metrics_txt = (f"EAR:{ear:.2f}(s:{ear_smooth:.2f})  MAR:{mar:.2f}  "
                       f"ROLL:{roll:.1f}d  "
                       f"CNN_E:{eye_pred:.2f}  CNN_M:{mouth_pred:.2f}")
        cv2.putText(frame, metrics_txt, (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

        # --- Red border on DANGER ---
        if status == "DANGER":
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

    else:
        # ---- No face detected ----
        cv2.rectangle(frame, (0, 0), (w, 55), (18, 18, 18), -1)
        cv2.putText(frame, "No face detected", (12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2, cv2.LINE_AA)
        # Show last known values faded
        cv2.putText(frame,
                    f"Last — EAR:{last_ear:.2f}  MAR:{last_mar:.2f}  Fatigue:{last_fatigue_pct}%",
                    (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1, cv2.LINE_AA)

    # ---- Show frame ----
    cv2.imshow(_win_title, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================
# CLEANUP
# ============================================================
stop_alarm()
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
pygame.mixer.quit()
print("✅  System stopped cleanly.")
print(f"    Total driving time : {int(time.time() - start_time) // 60} min "
      f"{int(time.time() - start_time) % 60} sec")
print(f"    Yawns detected     : {yawn_count}")
print(f"    Eye closures       : {eye_closure_count}")
