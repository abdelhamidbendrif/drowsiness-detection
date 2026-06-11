# DriveGuard - Speaker Notes

## Slide 1 - Cover
Introduce the project title, the team members, the supervisor and the academic context.

## Slide 2 - Presentation Outline
Explain that the presentation moves from the problem to the implemented real-time solution, then to validation, limitations and future work.

## Slide 3 - Motivation
Driver drowsiness is dangerous because it reduces reaction time and attention. A robust solution should consider several visual cues, not only eye closure.

## Slide 4 - Problem Statement
The objective is to analyze a live camera stream and classify the driver state as normal, warning or danger while preserving real-time behavior.

## Slide 5 - Objectives and Contributions
Highlight the three contributions: deep learning detection, multimodal temporal decision logic and a complete operational interface with alerts.

## Slide 6 - Global System Architecture
Describe the pipeline: camera input, MediaPipe landmarks, CNN inference, fusion, decision and outputs.

## Slide 7 - Datasets and Training Strategy
Present the MRL Eye dataset for eye state and the yawn dataset for mouth state. Be transparent that the mouth model still needs a stronger independent test split.

## Slide 8 - CNN Model Design
Explain why MobileNetV2 was chosen: lightweight, suitable for transfer learning and realistic for CPU-based real-time inference.

## Slide 9 - Facial Geometry Features
Define EAR, MAR and head pose. Emphasize that these measurements make the system more interpretable than a pure black-box classifier.

## Slide 10 - Multimodal Fusion
Explain the fusion formula and why eyes receive the largest weight while mouth and head pose provide complementary evidence.

## Slide 11 - Calibration and Temporal Reliability
Explain that calibration adapts thresholds to the driver and that time-based logic makes the system stable across different FPS values.

## Slide 12 - Decision Logic and Alerts
Show how normal, warning and danger states are selected. Mention sustained danger and release hysteresis to avoid flickering.

## Slide 13 - Evidence and Notification Pipeline
Explain that a confirmed danger event creates useful evidence: alarm, screenshot, video, CSV log, dashboard event and optional Telegram notification.

## Slide 14 - Real-Time Dashboard
Show the dashboard as the control center. Mention that the final version updates directly in the browser every 200 ms without Streamlit reruns.

## Slide 15 - Implementation Details
Map the main files to their responsibilities: detector, dashboard, training scripts, models and documentation.

## Slide 16 - Validation and Final Checks
Summarize what was verified: syntax, dependencies, model loading, dashboard rendering, JavaScript engine and GitHub upload.

## Slide 17 - Limitations
Be honest: the mouth model needs stronger validation, real-world conditions are difficult and the system uses temporal smoothing, not a true LSTM.

## Slide 18 - Future Work
Present concrete improvements: independent splits, full-system evaluation, optimized fusion, true temporal models and TensorFlow Lite deployment.

## Slide 19 - Demonstration Scenario
Use this slide before the live demo. Follow the steps exactly: dashboard, detector, calibration, normal behavior, yawn/eye closure and evidence.

## Slide 20 - Conclusion
Conclude that DriveGuard is a complete academic prototype combining deep learning, facial geometry, real-time decision logic and a professional monitoring interface.
