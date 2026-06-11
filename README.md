# DriveGuard - Multimodal Driver Drowsiness Detection

DriveGuard is a real-time prototype that estimates driver fatigue from a camera stream. It combines neural-network predictions with facial geometry, personal calibration, temporal smoothing, contextual risk, and graduated alerts.

## What the project implements

- Real-time capture from an integrated webcam, virtual camera, or phone IP stream.
- MediaPipe Face Mesh landmarks for eye, mouth, iris, and head measurements.
- Two MobileNetV2 binary classifiers:
  - `eye_model.keras`: eye state prediction.
  - `mouth_model.keras`: yawn prediction.
- Geometric features:
  - Eye Aspect Ratio (EAR).
  - Mouth Aspect Ratio (MAR).
  - Approximate head roll and pitch.
- Multimodal fatigue fusion:
  - Eyes: 50 percent.
  - Mouth: 30 percent.
  - Head pose: 20 percent.
- Personal EAR/MAR calibration at the beginning of each session.
- Time-based smoothing and event detection, independent of camera FPS.
- A 60-second fatigue trend used for predictive pre-alerts.
- Context multipliers for high-risk hours and long driving duration.
- Three states: `NORMAL`, `WARNING`, and `DANGER`.
- Sound alarm, screenshot, incident video, CSV session log, and optional Telegram notification.
- Streamlit dashboard with partial refreshes for smooth real-time monitoring.
- Optional iris-attention heatmap in the OpenCV window.

## Important scientific note

The current prototype does **not** contain a recurrent LSTM layer. It uses a time-based smoothing window over recent multimodal scores. In the report and presentation, describe it as:

> A hybrid multimodal system combining MobileNetV2 classifiers, MediaPipe facial geometry, adaptive calibration, temporal smoothing, and trend analysis.

A true CNN-LSTM sequence model remains future work.

## Project structure

| Path | Purpose |
| --- | --- |
| `main.py` | Camera capture, inference, fusion, alerts, recording, and OpenCV interface |
| `dashboard.py` | Stable Streamlit shell and local state endpoint |
| `dashboard_live.html` | Zero-rerun dashboard updated directly in the browser |
| `deep-learning.ipynb` | Kaggle notebook used to train and evaluate the eye model and original mouth model |
| `models/eye_model.keras` | Trained eye classifier |
| `models/mouth_model.keras` | Trained yawn classifier |
| `train_mouth_model_v2.py` | Kaggle training script for the mouth model |
| `PROJECT_FULL_REVIEW.md` | Complete technical, scientific, and operational project review |
| `alarm.wav` | Alarm sound; regenerated automatically if missing |
| `outputs/` | Screenshots, videos, CSV logs, and dashboard state |
| `SMARTPHONE_WEBCAM.md` | Phone-camera setup instructions |

The original training datasets are not included in this repository. The notebook references their Kaggle paths.

## Installation

The tested environment uses Python 3.12.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Training-only dependencies are separate:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-training.txt
```

Make sure both `.keras` files are present in `models/`. The folder is ignored by Git, so include it manually in the final ZIP or configure Git LFS before repository submission.

## Run the detector

Integrated webcam:

```powershell
.\venv\Scripts\python.exe main.py --source 0
```

DroidCam or another virtual camera:

```powershell
.\venv\Scripts\python.exe main.py --source 1
```

Android IP Webcam:

```powershell
.\venv\Scripts\python.exe main.py --source phone 192.168.1.42
```

Generic stream URL:

```powershell
.\venv\Scripts\python.exe main.py --source url http://192.168.1.42:8080/video
```

Shorter calibration for a rehearsal:

```powershell
.\venv\Scripts\python.exe main.py --source 0 --calibration-seconds 10
```

For the final evaluation, keep the default 30-second calibration when possible.

## Run the dashboard

Start it in a second terminal:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard.py
```

The visible dashboard is loaded only once. JavaScript polls the local state endpoint every 0.2 seconds and changes only the text nodes, SVG attributes, and bar widths whose values changed. There are no Streamlit fragments, page reruns, or Plotly component reconstructions during live monitoring.

## Camera-window controls

- `H`: toggle the iris-attention heatmap.
- `Q`: stop the session and save the alert log.

## Outputs

- `outputs/alert_DANGER_<timestamp>.jpg`: danger screenshot.
- `outputs/videos/incident_<timestamp>.avi`: five seconds before and after an alert.
- `outputs/session_<timestamp>.csv`: alert events for one session.
- `outputs/state.json`: atomic live state consumed by the dashboard.

## Detection logic

1. MediaPipe detects one face and its landmarks.
2. The system calculates EAR, MAR, and approximate head pose.
3. Eye and face crops are classified by the two MobileNetV2 models.
4. CNN and geometric scores are fused for eyes and mouth.
5. Eye, mouth, and head risks are combined into a multimodal score.
6. The score is smoothed over a 1.5-second window.
7. Time-of-day and driving-duration context modifies the final score.
8. Trend analysis can create a predictive pre-alert.
9. A calibrated danger state sustained for 1.5 seconds triggers the alarm and incident capture.

## Limitations

- No true LSTM sequence model is included.
- The available training script covers only the mouth model.
- Full validation on MRL Eye, UTA-RLDD, and NTHU-DDD is not included.
- Head pose is approximate rather than a full solvePnP estimate.
- The iris heatmap represents iris position, not a calibrated gaze vector.
- Performance depends on lighting, camera angle, glasses, and phone-network quality.
- Calibration assumes the driver starts the session in an alert state.

## Recommended presentation flow

1. Start the dashboard.
2. Start the detector and complete calibration while looking forward.
3. Demonstrate normal eye and mouth values.
4. Demonstrate a yawn and a prolonged eye closure.
5. Show the dashboard trend and alert history.
6. Show the generated screenshot, incident video, and session CSV.
7. Present the true CNN-LSTM model and broader dataset evaluation as future work.
