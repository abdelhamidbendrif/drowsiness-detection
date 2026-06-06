import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pygame
import time
import os
from scipy.spatial import distance

# ============================================================
# INITIALISATION
# ============================================================
pygame.mixer.init()

print("⏳ Chargement des modèles...")
eye_model   = tf.keras.models.load_model('models/eye_model.keras')
mouth_model = tf.keras.models.load_model('models/mouth_model.keras')
print("✅ Modèles chargés !")

# Nouvelle API MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# LANDMARKS
# ============================================================
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]

# ============================================================
# FONCTIONS
# ============================================================
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth_indices]
    A = distance.euclidean(pts[2], pts[6])
    B = distance.euclidean(pts[3], pts[7])
    C = distance.euclidean(pts[0], pts[1])
    return (A + B) / (2.0 * C)

def get_head_tilt(landmarks, w, h):
    left_eye  = (landmarks[133].x * w, landmarks[133].y * h)
    right_eye = (landmarks[362].x * w, landmarks[362].y * h)
    nose      = (landmarks[1].x * w,   landmarks[1].y * h)
    chin      = (landmarks[152].x * w, landmarks[152].y * h)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    roll  = abs(np.degrees(np.arctan2(dy, dx)))
    pitch = abs((chin[1] - nose[1]) / (h * 0.1))
    return roll, pitch

def preprocess(frame):
    img = cv2.resize(frame, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def play_alarm():
    try:
        pygame.mixer.music.load('alarm.wav')
        pygame.mixer.music.play()
    except:
        print("\a")

def save_screenshot(frame, level):
    os.makedirs('outputs', exist_ok=True)
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = f'outputs/alert_{level}_{ts}.jpg'
    cv2.imwrite(path, frame)
    print(f"📸 Screenshot : {path}")

# ============================================================
# VARIABLES
# ============================================================
alert_sent = False
start_time = time.time()

EAR_THRESHOLD  = 0.25
MAR_THRESHOLD  = 0.6

# ============================================================
# BOUCLE PRINCIPALE
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🎥 Système démarré — Appuie sur 'q' pour quitter")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Métriques géométriques
        ear = (eye_aspect_ratio(landmarks, LEFT_EYE,  w, h) +
               eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)) / 2.0
        mar        = mouth_aspect_ratio(landmarks, MOUTH, w, h)
        roll, pitch = get_head_tilt(landmarks, w, h)

        # Prédictions CNN
        eye_pred   = float(eye_model.predict(preprocess(frame),   verbose=0)[0][0])
        mouth_pred = float(mouth_model.predict(preprocess(frame), verbose=0)[0][0])

        # Scores combinés
        ear_score   = max(0, (EAR_THRESHOLD - ear) / EAR_THRESHOLD)
        mar_score   = max(0, (mar - MAR_THRESHOLD) / MAR_THRESHOLD)
        eye_final   = 0.5 * eye_pred   + 0.5 * ear_score
        mouth_final = 0.5 * mouth_pred + 0.5 * mar_score
        head_final  = min(1.0, roll / 30.0 + pitch / 5.0)

        # Score fatigue final
        fatigue = 0.50 * eye_final + 0.30 * mouth_final + 0.20 * head_final
        fatigue_pct = int(fatigue * 100)

        # Niveau alerte
        if fatigue_pct < 30:
            status, color = "NORMAL",  (0, 255, 0)
        elif fatigue_pct < 60:
            status, color = "WARNING", (0, 165, 255)
        else:
            status, color = "DANGER",  (0, 0, 255)
            if not alert_sent:
                play_alarm()
                save_screenshot(frame, "DANGER")
                alert_sent = True

        if fatigue_pct < 60:
            alert_sent = False

        # ---- AFFICHAGE ----
        # Bande noire en haut
        cv2.rectangle(frame, (0, 0), (w, 95), (20, 20, 20), -1)

        # Status
        cv2.putText(frame, f"STATUS: {status}", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Score + barre
        cv2.putText(frame, f"Fatigue: {fatigue_pct}%", (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        bar_w = int((w - 20) * fatigue)
        cv2.rectangle(frame, (10, 80), (w-10, 90), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 80), (10 + bar_w, 90), color, -1)

        # Métriques bas
        cv2.putText(frame,
                    f"EAR:{ear:.2f}  MAR:{mar:.2f}  HEAD:{roll:.1f}deg  CNN_Eye:{eye_pred:.2f}  CNN_Mouth:{mouth_pred:.2f}",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Durée
        elapsed    = int(time.time() - start_time)
        mins, secs = divmod(elapsed, 60)
        cv2.putText(frame, f"{mins:02d}:{secs:02d}", (w - 80, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        # Bordure rouge si DANGER
        if status == "DANGER":
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 5)

    else:
        cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.putText(frame, "Aucun visage detecte", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    cv2.imshow("Drowsiness Detection System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Système arrêté.")
