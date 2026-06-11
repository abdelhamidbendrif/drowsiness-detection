"""
main.py v4 — Real-Time Multimodal Driver Drowsiness Detection
=============================================================
NEW IN V4:
  [FIX]    False-alarm fix — stronger debounce, softer thresholds, hysteresis
  [NEW-1]  Temporal smoothing — fatigue averaged over recent frames
  [NEW-2]  Telegram bot — sends alert photo + message to your phone
  [NEW-3]  Iris-attention heatmap — visualizes recent iris positions
  [NEW-4]  Shared JSON state file — feeds the Streamlit dashboard in real time
  [NEW-5]  Attention heatmap overlay on the video frame

Usage:
  python main.py
  python main.py --source url http://192.168.1.9:8080/video
  python main.py --source phone 192.168.1.9
  python main.py --telegram-token TOKEN --telegram-chat CHAT_ID
"""

import argparse, threading, os, csv, collections, datetime, json, time

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="Real-time driver drowsiness detector")
parser.add_argument(
    "--source", nargs="+", default=["0"],
    help="Camera index, 'url <stream-url>', or 'phone <phone-ip>'"
)
parser.add_argument("--phone-port", type=int, default=8080, help="IP Webcam port")
parser.add_argument(
    "--calibration-seconds", type=float, default=30.0,
    help="Personal calibration duration (minimum 5 seconds)"
)
parser.add_argument("--telegram-token", default="", help="Telegram bot token")
parser.add_argument("--telegram-chat",  default="", help="Telegram chat ID")
args = parser.parse_args()

# Delay heavy imports until after CLI parsing so --help is immediate.
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
try:
    _ = tf.keras.models
except AttributeError:
    import keras as _keras; tf.keras = _keras
import pygame
from scipy.spatial import distance
from scipy.io import wavfile

try:
    import requests as _requests
    TELEGRAM_AVAILABLE = True
except Exception:
    TELEGRAM_AVAILABLE = False

TELEGRAM_TOKEN   = args.telegram_token
TELEGRAM_CHAT_ID = args.telegram_chat
USE_TELEGRAM     = TELEGRAM_AVAILABLE and bool(TELEGRAM_TOKEN) and bool(TELEGRAM_CHAT_ID)

_src = args.source
if _src[0] == "phone":
    if len(_src) < 2:
        parser.error("--source phone requires the phone IP, for example: --source phone 192.168.1.42")
    CAMERA_SOURCE = f"http://{_src[1]}:{args.phone_port}/video"; IS_IP_CAMERA = True
elif len(_src) == 2 and _src[0] == "url":
    CAMERA_SOURCE = _src[1]; IS_IP_CAMERA = True
else:
    try:    CAMERA_SOURCE = int(_src[0])
    except: CAMERA_SOURCE = _src[0]
    IS_IP_CAMERA = False

print(f"[CAMERA] {CAMERA_SOURCE} | Telegram: {'ON' if USE_TELEGRAM else 'OFF'}")

# ============================================================
# CONSTANTS
# ============================================================
DISPLAY_W, DISPLAY_H = 640, 480
SKIP_FRAMES   = 8 if IS_IP_CAMERA else 5
CALIB_SECONDS = max(5.0, args.calibration_seconds)
ALARM_WAV     = "alarm.wav"
SESSION_ID    = time.strftime("%Y%m%d_%H%M%S")

# ---- Fusion weights ----
W_EYE, W_MOUTH, W_HEAD   = 0.50, 0.30, 0.20
W_CNN_EYE,  W_GEO_EYE    = 0.70, 0.30
W_CNN_MOUTH, W_GEO_MOUTH = 0.50, 0.50
EAR_SMOOTH_ALPHA          = 0.4
SMOOTH_ALPHA              = 0.25   # softer smoothing → less jumpy score

# ---- FIX: Safer default thresholds ----
EAR_THRESHOLD_DEF = 0.22   # was 0.25 — less sensitive
MAR_THRESHOLD_DEF = 0.65   # was 0.60 — less sensitive
EYE_CLOSE_MIN_SEC = 1.0
YAWN_MIN_SEC      = 1.2
BLINK_MIN_SEC     = 0.08
BLINK_MAX_SEC     = 0.80
WARNING_THRESHOLD = 28
DANGER_THRESHOLD  = 58

# ---- Time-based debounce (stable across camera FPS) ----
DANGER_CONFIRM_SEC     = 1.5
DANGER_RELEASE_SEC     = 1.0
PRE_ALERT_COOLDOWN_SEC = 30.0

# ---- Temporal smoothing buffer ----
TEMPORAL_WINDOW_SEC = 1.5
temporal_buffer = collections.deque()

# ---- Environmental ----
HIGH_RISK_HOURS   = list(range(2, 6))
DRIVE_WARN_SEC    = 2 * 3600
DRIVE_DANGER_SEC  = 4 * 3600

# ---- Recording ----
RECORD_PRE_SEC, RECORD_POST_SEC, RECORD_FPS = 5, 5, 10
_frame_buffer    = collections.deque(maxlen=int(RECORD_PRE_SEC * RECORD_FPS))
_recording       = False
_rec_writer      = None
_rec_frames_left = 0
_rec_last_write  = 0.0
_last_buffer_sample = 0.0
_rec_lock        = threading.Lock()

# ---- Trend ----
TREND_WINDOW_SEC  = 60
TREND_SAMPLE_SEC  = 0.5
TREND_RISE_THRESH = 0.30   # was 0.25 — less trigger-happy
TREND_SAMPLES     = collections.deque(maxlen=int(TREND_WINDOW_SEC/TREND_SAMPLE_SEC)+5)

# ---- Iris-attention heatmap ----
_heatmap          = np.zeros((DISPLAY_H, DISPLAY_W), dtype=np.float32)
HEATMAP_DECAY     = 0.98   # each frame: heatmap * decay
HEATMAP_ALPHA     = 0.35   # blend opacity
HEATMAP_RADIUS    = 30
_hy,_hx = np.ogrid[-HEATMAP_RADIUS:HEATMAP_RADIUS+1,
                    -HEATMAP_RADIUS:HEATMAP_RADIUS+1]
_heatmap_kernel = np.clip(
    1.0-np.sqrt(_hx*_hx+_hy*_hy)/HEATMAP_RADIUS, 0.0, 1.0
).astype(np.float32)

# ---- Shared state for dashboard ----
STATE_FILE        = "outputs/state.json"

# Landmarks
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]
LEFT_EYE_DRAW  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE_DRAW = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
MOUTH_DRAW     = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# ============================================================
# ALARM
# ============================================================
def _generate_alarm_wav(path, duration=3.0, sr=44100):
    pulse=int(sr*0.25); fade=int(sr*0.05); n=int(sr*duration)
    sig=np.zeros(n,dtype=np.float32)
    for i,s in enumerate(range(0,n,pulse)):
        e=min(s+pulse,n); seg=np.arange(e-s)
        freq=880 if i%2==0 else 440
        wave=np.sin(2*np.pi*freq*seg/sr).astype(np.float32)
        env=np.ones(len(wave),dtype=np.float32); f=min(fade,len(wave)//2)
        env[:f]=np.linspace(0,1,f); env[-f:]=np.linspace(1,0,f)
        sig[s:e]=wave*env
    peak=np.max(np.abs(sig)) or 1.0
    wavfile.write(path,sr,(sig/peak*0.9*32767).astype(np.int16))

# ============================================================
# INIT
# ============================================================
print("[INIT] Initialising...")
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    if not os.path.isfile(ALARM_WAV): _generate_alarm_wav(ALARM_WAV)
    alarm_sound=pygame.mixer.Sound(ALARM_WAV); alarm_channel=None
    print("[OK] Alarm loaded.")
except Exception as e:
    alarm_sound=alarm_channel=None; print(f"[WARN] Alarm: {e}")

print("[INIT] Loading CNN models...")
try:
    eye_model   = tf.keras.models.load_model('models/eye_model.keras')
    mouth_model = tf.keras.models.load_model('models/mouth_model.keras')
    print("[OK] Models loaded.")
except Exception as e:
    print(f"[ERROR] Model failed: {e}"); raise SystemExit(1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/videos', exist_ok=True)

# ============================================================
# HELPERS
# ============================================================
def ear(lm,idx,w,h):
    pts=[(lm[i].x*w,lm[i].y*h) for i in idx]
    A=distance.euclidean(pts[1],pts[5]); B=distance.euclidean(pts[2],pts[4])
    C=distance.euclidean(pts[0],pts[3]); return (A+B)/(2*C+1e-6)

def mar(lm,idx,w,h):
    pts=[(lm[i].x*w,lm[i].y*h) for i in idx]
    A=distance.euclidean(pts[2],pts[6]); B=distance.euclidean(pts[3],pts[7])
    C=distance.euclidean(pts[0],pts[1]); return (A+B)/(2*C+1e-6)

def head_tilt(lm,w,h):
    le=(lm[133].x*w,lm[133].y*h); re=(lm[362].x*w,lm[362].y*h)
    nose=(lm[1].x*w,lm[1].y*h); chin=(lm[152].x*w,lm[152].y*h)
    roll=abs(np.degrees(np.arctan2(re[1]-le[1],re[0]-le[0])))
    pitch=abs((chin[1]-nose[1])/(h*0.1+1e-6)); return roll,pitch

def prep_frame(f):
    img=cv2.resize(f,(64,64)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype(np.float32)/255.0,axis=0)

def prep_eye(f,lm,w,h):
    idx=LEFT_EYE+RIGHT_EYE
    xs=[int(lm[i].x*w) for i in idx]; ys=[int(lm[i].y*h) for i in idx]
    p=20; x1,y1=max(0,min(xs)-p),max(0,min(ys)-p)
    x2,y2=min(w,max(xs)+p),min(h,max(ys)+p)
    crop=f[y1:y2,x1:x2]
    if crop.size==0: return prep_frame(f)
    img=cv2.resize(crop,(64,64)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype(np.float32)/255.0,axis=0)

def prep_face(f,lm,w,h):
    fidx=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,
          379,378,400,377,152,148,176,149,150,136,172,58,132,93,
          234,127,162,21,54,103,67,109]
    xs=[int(lm[i].x*w) for i in fidx]; ys=[int(lm[i].y*h) for i in fidx]
    p=10; x1,y1=max(0,min(xs)-p),max(0,min(ys)-p)
    x2,y2=min(w,max(xs)+p),min(h,max(ys)+p)
    crop=f[y1:y2,x1:x2]
    if crop.size==0: return prep_frame(f)
    img=cv2.resize(crop,(64,64)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype(np.float32)/255.0,axis=0)

def icolor(pct):
    if pct<30: r=int(pct/30*200); return (0,200-r,r)
    elif pct<60: t=(pct-30)/30; return (0,int((1-t)*165),90+int(t*255)//2)
    return (0,0,255)

def overlay_rect(frame, p1, p2, color=(15,23,42), alpha=0.86, border=None):
    """Draw a translucent panel while preserving the camera image."""
    layer=frame.copy()
    cv2.rectangle(layer,p1,p2,color,-1)
    cv2.addWeighted(layer,alpha,frame,1-alpha,0,frame)
    if border is not None:
        cv2.rectangle(frame,p1,p2,border,1,cv2.LINE_AA)

def draw_progress(frame,x1,y1,x2,y2,value,color,markers=()):
    value=max(0.0,min(1.0,float(value)))
    cv2.rectangle(frame,(x1,y1),(x2,y2),(48,58,72),-1)
    if value>0:
        cv2.rectangle(frame,(x1,y1),(x1+int((x2-x1)*value),y2),color,-1)
    for marker in markers:
        mx=x1+int((x2-x1)*marker)
        cv2.line(frame,(mx,y1-2),(mx,y2+2),(210,220,230),1,cv2.LINE_AA)

def draw_outline(frame,lm,idx,w,h,color,t=1):
    pts=np.array([(int(lm[i].x*w),int(lm[i].y*h)) for i in idx],dtype=np.int32)
    cv2.polylines(frame,[pts],True,color,t)

def save_screenshot(frame,level):
    ts=time.strftime("%Y%m%d_%H%M%S"); path=f'outputs/alert_{level}_{ts}.jpg'
    cv2.imwrite(path,frame); return path

def start_alarm():
    global alarm_channel
    if alarm_sound and (alarm_channel is None or not alarm_channel.get_busy()):
        alarm_channel=alarm_sound.play(loops=-1)

def stop_alarm():
    global alarm_channel
    if alarm_channel: alarm_channel.stop(); alarm_channel=None

# ============================================================
# NEW: TELEGRAM
# ============================================================
def send_telegram(msg: str, image_path: str = None):
    if not USE_TELEGRAM: return
    def _send():
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
            _requests.post(f"{url}/sendMessage",
                           data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
            if image_path and os.path.isfile(image_path):
                with open(image_path, "rb") as img:
                    _requests.post(f"{url}/sendPhoto",
                                   data={"chat_id": TELEGRAM_CHAT_ID},
                                   files={"photo": img}, timeout=10)
        except Exception as e:
            print(f"[WARN] Telegram error: {e}")
    threading.Thread(target=_send, daemon=True).start()

# ============================================================
# IRIS ATTENTION HEATMAP
# ============================================================
def update_heatmap(lm, w, h):
    global _heatmap
    try:
        # Get iris centre (average of iris landmarks)
        lx = int(np.mean([lm[i].x for i in LEFT_IRIS])  * w)
        ly = int(np.mean([lm[i].y for i in LEFT_IRIS])  * h)
        rx = int(np.mean([lm[i].x for i in RIGHT_IRIS]) * w)
        ry = int(np.mean([lm[i].y for i in RIGHT_IRIS]) * h)
        gx = (lx + rx) // 2;  gy = (ly + ry) // 2
        _heatmap *= HEATMAP_DECAY
        r = HEATMAP_RADIUS
        x1,y1 = max(0,gx-r), max(0,gy-r)
        x2,y2 = min(w,gx+r+1), min(h,gy+r+1)
        kx1,ky1 = x1-(gx-r), y1-(gy-r)
        kx2,ky2 = kx1+(x2-x1), ky1+(y2-y1)
        _heatmap[y1:y2,x1:x2] += _heatmap_kernel[ky1:ky2,kx1:kx2]
    except Exception:
        pass

def draw_heatmap(frame):
    if _heatmap.max() < 0.01: return frame
    norm = np.clip(_heatmap / (_heatmap.max() + 1e-6), 0, 1)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = (norm > 0.05).astype(np.float32)
    mask3 = np.stack([mask]*3, axis=2)
    blended = (frame.astype(np.float32) * (1 - mask3 * HEATMAP_ALPHA) +
               colored.astype(np.float32) * mask3 * HEATMAP_ALPHA).astype(np.uint8)
    return blended

# ============================================================
# ENVIRONMENTAL
# ============================================================
def env_mult(drive_sec):
    h = datetime.datetime.now().hour
    hf = 2.0 if h in HIGH_RISK_HOURS else 1.0
    if   drive_sec >= DRIVE_DANGER_SEC: df,dl = 2.0, f">4h ×2"
    elif drive_sec >= DRIVE_WARN_SEC:   df,dl = 1.5, f">2h ×1.5"
    else:                               df,dl = 1.0, ""
    hl = f"{h:02d}h ×2" if h in HIGH_RISK_HOURS else ""
    parts = [x for x in [hl,dl] if x]
    return min(hf*df, 3.0), ("  ".join(parts) if parts else "Normal")

# ============================================================
# TREND
# ============================================================
def compute_trend(samples):
    if len(samples) < 10: return 0.0
    now_t=time.time(); pts=[(t,v) for t,v in samples if t>=now_t-TREND_WINDOW_SEC]
    if len(pts) < 5: return 0.0
    ts=np.array([p[0] for p in pts]); vs=np.array([p[1] for p in pts])
    ts=(ts-ts[0])/60.0
    return float(np.polyfit(ts,vs,1)[0]) if ts[-1]>0 else 0.0

# ============================================================
# RECORDING
# ============================================================
def start_recording(pre_frames):
    global _recording,_rec_writer,_rec_frames_left,_rec_last_write
    with _rec_lock:
        if _recording: return
        ts=time.strftime("%Y%m%d_%H%M%S"); path=f"outputs/videos/incident_{ts}.avi"
        _rec_writer=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'XVID'),
                                    RECORD_FPS,(DISPLAY_W,DISPLAY_H))
        if not _rec_writer.isOpened():
            _rec_writer.release(); _rec_writer=None
            print(f"[WARN] Cannot start incident recording: {path}")
            return
        for f in pre_frames: _rec_writer.write(f)
        _rec_frames_left=int(RECORD_POST_SEC*RECORD_FPS)
        _rec_last_write=time.time(); _recording=True
        print(f"[REC] Recording -> {path}")

def feed_recording(frame):
    global _recording,_rec_writer,_rec_frames_left,_rec_last_write
    with _rec_lock:
        if not _recording: return
        now_ts=time.time()
        if now_ts-_rec_last_write < 1.0/RECORD_FPS: return
        _rec_last_write=now_ts
        _rec_writer.write(frame); _rec_frames_left-=1
        if _rec_frames_left<=0:
            _rec_writer.release(); _rec_writer=None; _recording=False
            print("[REC] Recording saved.")

# ============================================================
# SHARED STATE → Dashboard
# ============================================================
_state_lock = threading.Lock()
_state_last_write = 0.0
_state = {
    "fatigue_pct": 0, "status": "NORMAL", "ear": 0.0, "mar": 0.0,
    "roll": 0.0, "pitch": 0.0, "blink_rate": 0.0,
    "eye_pred": 0.0, "mouth_pred": 0.0,
    "env_mult": 1.0, "env_label": "Normal",
    "yawn_count": 0, "eye_closure_count": 0,
    "drive_min": 0.0, "fps": 0.0, "trend_slope": 0.0,
    "calibrated": False, "calibration_progress": 0,
    "ear_threshold": EAR_THRESHOLD_DEF, "mar_threshold": MAR_THRESHOLD_DEF,
    "warning_threshold": WARNING_THRESHOLD, "danger_threshold": DANGER_THRESHOLD,
    "face_detected": False, "alarm_active": False, "recording": False,
    "telegram_enabled": USE_TELEGRAM,
    "alert_log": [], "fatigue_history": [], "timestamp": "", "updated_at": 0.0
}

def update_state(**kwargs):
    global _state_last_write
    now_ts=time.time()
    with _state_lock:
        _state.update(kwargs)
        _state["timestamp"] = time.strftime("%H:%M:%S")
        _state["updated_at"] = now_ts
        if now_ts-_state_last_write<0.2:
            return
        _state_last_write=now_ts
        snapshot=dict(_state)
    try:
        tmp_path=f"{STATE_FILE}.tmp"
        with open(tmp_path,"w",encoding="utf-8") as f:
            json.dump(snapshot,f)
        os.replace(tmp_path,STATE_FILE)
    except Exception:
        pass

# ============================================================
# SESSION LOG
# ============================================================
alert_log = []; fatigue_history = []

def log_alert(level, fatigue_pct, em, ts, drive_sec):
    entry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
             "level": level, "fatigue_pct": fatigue_pct,
             "env_mult": round(em,2), "drive_min": round(drive_sec/60,1)}
    alert_log.append(entry)
    with _state_lock:
        _state["alert_log"] = alert_log[-50:]
        _state["updated_at"] = time.time()
        snapshot=dict(_state)
    try:
        tmp_path=f"{STATE_FILE}.alert.tmp"
        with open(tmp_path,"w",encoding="utf-8") as f:
            json.dump(snapshot,f)
        os.replace(tmp_path,STATE_FILE)
    except Exception: pass

def save_log():
    if not alert_log: return
    path=f"outputs/session_{SESSION_ID}.csv"
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=alert_log[0].keys())
        w.writeheader(); w.writerows(alert_log)
    print(f"[LOG] Saved -> {path}")

# ============================================================
# STATE VARIABLES
# ============================================================
calib_ear_samples, calib_mar_samples = [], []
calibrated   = False; calib_start = time.time()
EAR_THRESHOLD = EAR_THRESHOLD_DEF
MAR_THRESHOLD = MAR_THRESHOLD_DEF

alert_sent = False; pre_alert_sent = False
danger_since = safe_since = None
last_pre_alert_time = -PRE_ALERT_COOLDOWN_SEC

yawn_count = eye_closure_count = 0
eye_close_started_at = mouth_open_started_at = None
yawn_active = eye_close_active = False

frame_counter = 0; eye_pred = mouth_pred = 0.0
fatigue_smooth = 0.0; ear_smooth = EAR_THRESHOLD_DEF
last_ear = last_mar = last_roll = 0.0; last_fatigue_pct = 0

# Blink rate with duration filtering
blink_started_at = None
blink_timestamps = collections.deque()
blink_rate = 0.0

fps_time = time.time(); fps_val = fps_count = 0
last_trend_sample = 0.0
start_time = time.time()

# ============================================================
# CAMERA
# ============================================================
if os.name=="nt" and isinstance(CAMERA_SOURCE,int):
    cap=cv2.VideoCapture(CAMERA_SOURCE,cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release(); cap=cv2.VideoCapture(CAMERA_SOURCE)
else:
    cap=cv2.VideoCapture(CAMERA_SOURCE)
if IS_IP_CAMERA:
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,480); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
else:
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,DISPLAY_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,DISPLAY_H); cap.set(cv2.CAP_PROP_FPS,30)

if not cap.isOpened():
    print(f"[ERROR] Cannot open camera: {CAMERA_SOURCE}"); raise SystemExit(1)

print("[START] Press 'q' to quit | Press 'h' to toggle iris heatmap")
_win = f"Drowsiness Detection v4  |  {'IP Cam' if IS_IP_CAMERA else f'Cam[{CAMERA_SOURCE}]'}"
cv2.namedWindow(_win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(_win, DISPLAY_W, DISPLAY_H)

show_heatmap = False   # toggle with 'h'

# ============================================================
# BACKGROUND THREADS
# ============================================================
_cap_frame=None; _cap_lock=threading.Lock(); _cap_running=True

def _cap_loop():
    global _cap_frame,_cap_running
    while _cap_running:
        ret,f=cap.read()
        if ret and f is not None:
            if f.shape[1]!=DISPLAY_W or f.shape[0]!=DISPLAY_H:
                f=cv2.resize(f,(DISPLAY_W,DISPLAY_H),interpolation=cv2.INTER_LINEAR)
            with _cap_lock: _cap_frame=f

if IS_IP_CAMERA:
    threading.Thread(target=_cap_loop,daemon=True).start()
    print("[THREAD] Capture thread started."); time.sleep(0.5)

_cnn_lock=threading.Lock(); _cnn_job=None; _cnn_result=(0.0,0.0); _cnn_running=True

def _cnn_loop():
    global _cnn_result,_cnn_running,_cnn_job
    while _cnn_running:
        with _cnn_lock: job=_cnn_job
        if job:
            ei,mi=job
            ep=float(eye_model(ei,training=False)[0][0])
            mp_v=float(mouth_model(mi,training=False)[0][0])
            with _cnn_lock:
                _cnn_result=(ep,mp_v)
                if _cnn_job is job: _cnn_job=None
        else: time.sleep(0.002)

threading.Thread(target=_cnn_loop,daemon=True).start()
print("[THREAD] CNN thread started.")

# ============================================================
# MAIN LOOP
# ============================================================
while True:
    if IS_IP_CAMERA:
        with _cap_lock: frame=_cap_frame
        if frame is None: time.sleep(0.01); continue
        frame=frame.copy()
    else:
        ret,frame=cap.read()
        if not ret: time.sleep(0.05); continue
        if frame.shape[1]!=DISPLAY_W or frame.shape[0]!=DISPLAY_H:
            frame=cv2.resize(frame,(DISPLAY_W,DISPLAY_H),interpolation=cv2.INTER_LINEAR)

    h,w=DISPLAY_H,DISPLAY_W
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    fps_count+=1; now=time.time()
    if now-fps_time>=1.0:
        fps_val=fps_count/(now-fps_time); fps_count=0; fps_time=now

    drive_elapsed=now-start_time
    if now-_last_buffer_sample>=1.0/RECORD_FPS:
        _frame_buffer.append(frame.copy())
        _last_buffer_sample=now

    em,el=env_mult(drive_elapsed)
    results=face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm=results.multi_face_landmarks[0].landmark

        ear_val=(ear(lm,LEFT_EYE,w,h)+ear(lm,RIGHT_EYE,w,h))/2.0
        mar_val=mar(lm,MOUTH,w,h)
        roll,pitch=head_tilt(lm,w,h)
        last_ear,last_mar,last_roll=ear_val,mar_val,roll

        # Calibration
        elapsed_calib=now-calib_start
        if not calibrated:
            calib_ear_samples.append(ear_val); calib_mar_samples.append(mar_val)
            if elapsed_calib>=CALIB_SECONDS:
                EAR_THRESHOLD=max(float(np.percentile(calib_ear_samples,12)),0.14)
                MAR_THRESHOLD=min(float(np.percentile(calib_mar_samples,88)),0.92)
                calibrated=True
                blink_started_at=eye_close_started_at=mouth_open_started_at=None
                eye_close_active=yawn_active=False
                print(f"[OK] Calibration done EAR={EAR_THRESHOLD:.3f} MAR={MAR_THRESHOLD:.3f}")

        # CNN
        frame_counter+=1
        if frame_counter%SKIP_FRAMES==0:
            with _cnn_lock: _cnn_job=(prep_eye(frame,lm,w,h),prep_face(frame,lm,w,h))
        with _cnn_lock: eye_pred,mouth_pred=_cnn_result

        # Fusion
        ear_smooth=EAR_SMOOTH_ALPHA*ear_val+(1-EAR_SMOOTH_ALPHA)*ear_smooth
        ear_score=max(0.0,(EAR_THRESHOLD-ear_smooth)/(EAR_THRESHOLD+1e-6))
        mar_score=max(0.0,(mar_val-MAR_THRESHOLD)/(MAR_THRESHOLD+1e-6))
        eye_f=W_CNN_EYE*eye_pred+W_GEO_EYE*ear_score
        mth_f=W_CNN_MOUTH*mouth_pred+W_GEO_MOUTH*mar_score
        hd_f=min(1.0,roll/30.0+pitch/5.0)
        raw=W_EYE*eye_f+W_MOUTH*mth_f+W_HEAD*hd_f

        # Time-based smoothing behaves consistently across camera frame rates
        temporal_buffer.append((now,raw))
        while temporal_buffer and now-temporal_buffer[0][0]>TEMPORAL_WINDOW_SEC:
            temporal_buffer.popleft()
        temporal_avg=float(np.mean([value for _,value in temporal_buffer]))

        fatigue_smooth=SMOOTH_ALPHA*temporal_avg+(1-SMOOTH_ALPHA)*fatigue_smooth
        fatigue_env=min(1.0,fatigue_smooth*em)
        fatigue_pct=int(fatigue_env*100)
        last_fatigue_pct=fatigue_pct

        if now-last_trend_sample>=TREND_SAMPLE_SEC:
            TREND_SAMPLES.append((now,fatigue_smooth))
            last_trend_sample=now
        trend_slope=compute_trend(TREND_SAMPLES)

        # Sample for history
        if len(fatigue_history)==0 or now-fatigue_history[-1][0]>=2.0:
            fatigue_history.append((now,fatigue_pct))
            if len(fatigue_history)>300: fatigue_history.pop(0)

        # Blink rate: count only short close/open cycles, not noisy crossings
        if ear_smooth<EAR_THRESHOLD:
            if blink_started_at is None: blink_started_at=now
        elif blink_started_at is not None:
            blink_duration=now-blink_started_at
            if BLINK_MIN_SEC<=blink_duration<=BLINK_MAX_SEC:
                blink_timestamps.append(now)
            blink_started_at=None
        while blink_timestamps and now-blink_timestamps[0]>60: blink_timestamps.popleft()
        blink_rate=len(blink_timestamps)

        # Time-based prolonged eye closure counter
        if ear_smooth<EAR_THRESHOLD:
            if eye_close_started_at is None: eye_close_started_at=now
            if now-eye_close_started_at>=EYE_CLOSE_MIN_SEC: eye_close_active=True
        else:
            if eye_close_active: eye_closure_count+=1
            eye_close_active=False; eye_close_started_at=None

        # Time-based yawn counter
        if mar_val>MAR_THRESHOLD:
            if mouth_open_started_at is None: mouth_open_started_at=now
            if now-mouth_open_started_at>=YAWN_MIN_SEC: yawn_active=True
        else:
            if yawn_active: yawn_count+=1
            yawn_active=False; mouth_open_started_at=None

        # Status — with hysteresis to prevent flickering
        if fatigue_pct < WARNING_THRESHOLD:   status="NORMAL"
        elif fatigue_pct < DANGER_THRESHOLD: status="WARNING"
        else:                  status="DANGER"

        # Pre-alert
        if (status=="WARNING" and trend_slope>=TREND_RISE_THRESH
                and not pre_alert_sent and calibrated
                and now-last_pre_alert_time>=PRE_ALERT_COOLDOWN_SEC):
            pre_alert_sent=True
            last_pre_alert_time=now
            log_alert("PRE-ALERT",fatigue_pct,em,now,drive_elapsed)
            print(f"[WARN] PRE-ALERT: +{trend_slope:.2f}/min")
        if status=="NORMAL": pre_alert_sent=False

        # Time-based hysteresis keeps behavior consistent at 8 or 30 FPS
        if status=="DANGER" and calibrated:
            if danger_since is None: danger_since=now
            safe_since=None
            if now-danger_since>=DANGER_CONFIRM_SEC and not alert_sent:
                start_alarm()
                img_path=save_screenshot(frame,"DANGER")
                start_recording(list(_frame_buffer))
                log_alert("DANGER",fatigue_pct,em,now,drive_elapsed)
                send_telegram(
                    f"🚨 DROWSINESS ALERT!\n"
                    f"Fatigue: {fatigue_pct}%\n"
                    f"Drive time: {int(drive_elapsed//60)} min\n"
                    f"ENV: ×{em:.1f}  {el}\n"
                    f"Time: {time.strftime('%H:%M:%S')}",
                    img_path
                )
                alert_sent=True
        else:
            danger_since=None
            if alert_sent:
                if safe_since is None: safe_since=now
                if now-safe_since>=DANGER_RELEASE_SEC:
                    stop_alarm(); alert_sent=False; safe_since=None
            else:
                safe_since=None

        feed_recording(frame)

        # Iris-attention heatmap
        update_heatmap(lm,w,h)

        # Landmarks
        lm_c=((0,255,0) if status=="NORMAL" else
              (0,165,255) if status=="WARNING" else (0,80,255))
        draw_outline(frame,lm,LEFT_EYE_DRAW,w,h,lm_c)
        draw_outline(frame,lm,RIGHT_EYE_DRAW,w,h,lm_c)
        draw_outline(frame,lm,MOUTH_DRAW,w,h,lm_c)

        bar_color=icolor(fatigue_pct)

        # Update dashboard state
        update_state(
            fatigue_pct=fatigue_pct, status=status,
            ear=round(ear_val,3), mar=round(mar_val,3),
            roll=round(roll,1), pitch=round(pitch,2),
            eye_pred=round(eye_pred,3), mouth_pred=round(mouth_pred,3),
            blink_rate=blink_rate, env_mult=em, env_label=el,
            yawn_count=yawn_count, eye_closure_count=eye_closure_count,
            drive_min=round(drive_elapsed/60,1), fps=round(fps_val,1),
            trend_slope=round(trend_slope,3),
            calibrated=calibrated,
            calibration_progress=min(100,int(elapsed_calib/CALIB_SECONDS*100)),
            ear_threshold=round(EAR_THRESHOLD,3), mar_threshold=round(MAR_THRESHOLD,3),
            warning_threshold=WARNING_THRESHOLD, danger_threshold=DANGER_THRESHOLD,
            face_detected=True, alarm_active=alert_sent, recording=_recording,
            telegram_enabled=USE_TELEGRAM,
            fatigue_history=[[t,v] for t,v in fatigue_history[-120:]]
        )

        # ============================================================
        # HEATMAP OVERLAY (toggle 'h')
        # ============================================================
        if show_heatmap:
            frame = draw_heatmap(frame)

        # ============================================================
        # UI
        # ============================================================
        status_color={"NORMAL":(80,220,145),"WARNING":(32,176,255),"DANGER":(90,75,255)}[status]

        # Top control bar
        overlay_rect(frame,(0,0),(w,104),(9,14,25),0.93)
        cv2.putText(frame,"DRIVER STATE",(16,22),cv2.FONT_HERSHEY_SIMPLEX,0.38,
                    (145,160,180),1,cv2.LINE_AA)
        cv2.putText(frame,status,(16,55),cv2.FONT_HERSHEY_SIMPLEX,0.95,
                    status_color,2,cv2.LINE_AA)
        cv2.putText(frame,f"Fatigue {fatigue_pct}%",(205,39),cv2.FONT_HERSHEY_SIMPLEX,0.66,
                    (235,240,248),2,cv2.LINE_AA)
        draw_progress(frame,205,54,w-112,68,fatigue_env,bar_color,
                      (WARNING_THRESHOLD/100,DANGER_THRESHOLD/100))
        bar_width=(w-112)-205
        warning_x=205+int(bar_width*WARNING_THRESHOLD/100)-6
        danger_x=205+int(bar_width*DANGER_THRESHOLD/100)-6
        cv2.putText(frame,"28",(warning_x,84),cv2.FONT_HERSHEY_SIMPLEX,0.32,
                    (145,160,180),1,cv2.LINE_AA)
        cv2.putText(frame,"58",(danger_x,84),cv2.FONT_HERSHEY_SIMPLEX,0.32,
                    (145,160,180),1,cv2.LINE_AA)

        e2=int(drive_elapsed); m2,s2=divmod(e2,60)
        cv2.putText(frame,"SESSION",(w-92,22),cv2.FONT_HERSHEY_SIMPLEX,0.34,
                    (145,160,180),1,cv2.LINE_AA)
        cv2.putText(frame,f"{m2:02d}:{s2:02d}",(w-102,50),cv2.FONT_HERSHEY_SIMPLEX,0.72,
                    (235,240,248),2,cv2.LINE_AA)
        cv2.putText(frame,f"{fps_val:.0f} FPS",(w-91,76),cv2.FONT_HERSHEY_SIMPLEX,0.38,
                    (145,160,180),1,cv2.LINE_AA)

        # Compact telemetry card
        panel_x=w-184
        overlay_rect(frame,(panel_x,118),(w-10,240),(12,20,34),0.86,(55,72,94))
        cv2.putText(frame,"LIVE SIGNALS",(panel_x+12,139),cv2.FONT_HERSHEY_SIMPLEX,0.36,
                    (145,160,180),1,cv2.LINE_AA)
        telemetry=[
            ("EAR",f"{ear_val:.3f}",ear_val<EAR_THRESHOLD),
            ("MAR",f"{mar_val:.3f}",mar_val>MAR_THRESHOLD),
            ("EYE CNN",f"{eye_pred:.2f}",eye_pred>0.6),
            ("MOUTH CNN",f"{mouth_pred:.2f}",mouth_pred>0.6),
        ]
        for i,(label,value,hot) in enumerate(telemetry):
            yy=162+i*19
            cv2.putText(frame,label,(panel_x+12,yy),cv2.FONT_HERSHEY_SIMPLEX,0.34,
                        (150,165,185),1,cv2.LINE_AA)
            cv2.putText(frame,value,(w-68,yy),cv2.FONT_HERSHEY_SIMPLEX,0.38,
                        ((75,90,255) if hot else (225,232,242)),1,cv2.LINE_AA)

        # Event summary and controls
        overlay_rect(frame,(10,h-78),(280,h-36),(12,20,34),0.84,(55,72,94))
        cv2.putText(frame,f"Yawns {yawn_count}   Closures {eye_closure_count}   Blinks/min {blink_rate:.0f}",
                    (20,h-60),cv2.FONT_HERSHEY_SIMPLEX,0.37,(220,228,238),1,cv2.LINE_AA)
        cv2.putText(frame,f"Trend {trend_slope:+.2f}/min   Context x{em:.1f}",
                    (20,h-43),cv2.FONT_HERSHEY_SIMPLEX,0.35,
                    ((32,176,255) if trend_slope>TREND_RISE_THRESH else (145,160,180)),1,cv2.LINE_AA)
        overlay_rect(frame,(0,h-28),(w,h),(9,14,25),0.94)
        cv2.putText(frame,"H  Heatmap",(10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.36,
                    (160,174,192),1,cv2.LINE_AA)
        cv2.putText(frame,"Q  End session",(104,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.36,
                    (160,174,192),1,cv2.LINE_AA)
        cv2.putText(frame,f"ENV x{em:.1f}  {el}",(w-125,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.34,
                    ((80,220,145) if em==1.0 else (32,176,255)),1,cv2.LINE_AA)

        # Calibration overlay
        if not calibrated:
            cp=min(100,int(elapsed_calib/CALIB_SECONDS*100))
            rs=max(0,int(CALIB_SECONDS-elapsed_calib))
            x1,y1,x2,y2=95,h//2-58,w-95,h//2+58
            overlay_rect(frame,(x1,y1),(x2,y2),(12,20,34),0.94,(56,189,248))
            cv2.putText(frame,"PERSONAL CALIBRATION",(x1+24,y1+28),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(248,189,56),2,cv2.LINE_AA)
            cv2.putText(frame,"Look at camera - eyes open - mouth closed",(x1+24,y1+53),
                        cv2.FONT_HERSHEY_SIMPLEX,0.39,(220,228,238),1,cv2.LINE_AA)
            cv2.putText(frame,f"{rs}s remaining",(x2-112,y1+28),
                        cv2.FONT_HERSHEY_SIMPLEX,0.37,(160,174,192),1,cv2.LINE_AA)
            draw_progress(frame,x1+24,y2-30,x2-24,y2-18,cp/100,(248,189,56))
            cv2.putText(frame,f"{cp}%",(x2-57,y2-38),cv2.FONT_HERSHEY_SIMPLEX,0.34,
                        (220,228,238),1,cv2.LINE_AA)

        # Banners
        if pre_alert_sent and status=="WARNING":
            overlay_rect(frame,(0,104),(w,136),(20,105,190),0.94)
            cv2.putText(frame,"FATIGUE TREND RISING - PLAN A SAFE BREAK",
                        (w//2-215,126),cv2.FONT_HERSHEY_SIMPLEX,0.49,(255,255,255),2,cv2.LINE_AA)

        if status=="DANGER":
            cv2.rectangle(frame,(0,0),(w-1,h-1),(0,0,255),6)

        if show_heatmap:
            cv2.putText(frame,"IRIS HEATMAP ON",(12,122),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,(80,230,195),1,cv2.LINE_AA)

        if _recording:
            cv2.circle(frame,(w-20,20),8,(0,0,255),-1)
            cv2.putText(frame,"REC",(w-50,25),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),1,cv2.LINE_AA)

    else:
        blink_started_at=eye_close_started_at=mouth_open_started_at=None
        eye_close_active=yawn_active=False
        update_state(
            face_detected=False, calibrated=calibrated,
            calibration_progress=min(100,int((now-calib_start)/CALIB_SECONDS*100)),
            drive_min=round(drive_elapsed/60,1), fps=round(fps_val,1),
            alarm_active=alert_sent, recording=_recording,
            ear_threshold=round(EAR_THRESHOLD,3), mar_threshold=round(MAR_THRESHOLD,3),
            warning_threshold=WARNING_THRESHOLD, danger_threshold=DANGER_THRESHOLD
        )
        overlay_rect(frame,(0,0),(w,76),(9,14,25),0.94)
        cv2.putText(frame,"DRIVER NOT VISIBLE",(18,34),cv2.FONT_HERSHEY_SIMPLEX,0.82,
                    (150,165,185),2,cv2.LINE_AA)
        cv2.putText(frame,"Reposition the camera or return to the frame",(19,58),
                    cv2.FONT_HERSHEY_SIMPLEX,0.40,(120,135,155),1,cv2.LINE_AA)
        overlay_rect(frame,(120,h//2-45),(w-120,h//2+45),(12,20,34),0.88,(55,72,94))
        cv2.putText(frame,"NO FACE DETECTED",(w//2-128,h//2-2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.72,(170,185,205),2,cv2.LINE_AA)
        cv2.putText(frame,f"Last fatigue score: {last_fatigue_pct}%",(w//2-92,h//2+28),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(125,140,160),1,cv2.LINE_AA)
        overlay_rect(frame,(0,h-28),(w,h),(9,14,25),0.94)
        cv2.putText(frame,"Q  End session",(12,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.36,
                    (160,174,192),1,cv2.LINE_AA)
        feed_recording(frame)

    cv2.imshow(_win, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('h'): show_heatmap = not show_heatmap

# ============================================================
# CLEANUP
# ============================================================
_cap_running=False; _cnn_running=False
stop_alarm()
if _rec_writer: _rec_writer.release()
cap.release(); cv2.destroyAllWindows(); face_mesh.close()
if pygame.mixer.get_init(): pygame.mixer.quit()
save_log()
total=int(time.time()-start_time)
print(f"[DONE] Drive: {total//60}m{total%60}s | Yawns:{yawn_count} | Closures:{eye_closure_count} | Alerts:{len(alert_log)}")
