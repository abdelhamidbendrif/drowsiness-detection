# 📱 Using Your Smartphone as a Webcam on Windows
## (for use with OpenCV / Python)

---

## Option 1 — DroidCam (Android / iPhone) ⭐ Recommended

**Why:** Free, stable, works as a proper virtual camera device — OpenCV sees it as `VideoCapture(1)` or `(2)`.

### Steps
1. **Phone:** Install **DroidCam** from Google Play / App Store.
2. **PC:** Install **DroidCam Windows Client** from `dev47apps.com`.
3. Connect via **USB** (best quality, no lag):
   - Enable **USB Debugging** on Android (`Settings → Developer Options`).
   - Open DroidCam client → select USB → Connect.
4. Or connect via **Wi-Fi**:
   - Open DroidCam on phone → note the IP address shown.
   - In the PC client enter that IP and port `4747`.

### Project usage
```powershell
# DroidCam registers as a virtual camera — usually index 1 or 2
.\venv\Scripts\python.exe main.py --source 1
```

---

## Option 2 — IP Webcam (Android only) — No PC client needed

**Why:** Streams MJPEG over Wi-Fi, readable directly by OpenCV via URL.

### Steps
1. **Phone:** Install **IP Webcam** (by Pavel Khlebovich) from Google Play.
2. Scroll to bottom → tap **Start Server**.
3. Note the URL shown, e.g. `http://192.168.1.42:8080`.

### Project usage
```powershell
# Replace the example address with the IP shown by IP Webcam
.\venv\Scripts\python.exe main.py --source phone 192.168.1.42

# Use this form if the application uses a different port
.\venv\Scripts\python.exe main.py --source phone 192.168.1.42 --phone-port 8080
```

> **Tip:** Phone and laptop must be on the **same Wi-Fi network**.
> For less latency use `http://{IP}:{PORT}/shot.jpg` in a loop (JPEG snapshots).

---

## Option 3 — EpocCam (iPhone) — Requires driver

**Why:** High-quality iOS camera, appears as a virtual webcam.

### Steps
1. **iPhone:** Install **EpocCam** from the App Store (Elgato).
2. **PC:** Install the **EpocCam driver** from `elgato.com/epoccam`.
3. Connect iPhone via USB or Wi-Fi.

### Project usage
```powershell
.\venv\Scripts\python.exe main.py --source 1
```

---

## Option 4 — OBS Virtual Camera (any phone via OBS)

Use OBS Studio + any phone's native camera app streamed via RTSP → OBS → Virtual Camera.
More complex to set up but gives maximum control over resolution/FPS.

---

## Comparison Table

| Solution       | OS     | Connection     | OpenCV index/URL            | Latency | Quality |
|----------------|--------|----------------|-----------------------------|---------|---------|
| DroidCam       | Android/iOS | USB or Wi-Fi | `VideoCapture(1)`           | Low     | Good    |
| IP Webcam      | Android | Wi-Fi         | `VideoCapture("http://...")` | Medium  | Good    |
| EpocCam        | iOS    | USB or Wi-Fi   | `VideoCapture(1)`           | Low     | Excellent|
| OBS Virtual Cam| Any    | USB/Wi-Fi/RTSP | `VideoCapture(1)`           | Medium  | Excellent|

---

## Discovering the Correct Index

```python
# Run this snippet to find all available camera indices
import cv2
for idx in range(5):
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        print(f"Camera found at index {idx}")
        cap.release()
```

---

## Fast demo command

Use a shorter calibration only for a presentation rehearsal:

```powershell
.\venv\Scripts\python.exe main.py --source phone 192.168.1.42 --calibration-seconds 10
```
