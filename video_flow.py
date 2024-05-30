import sys

import cv2
import mediapipe as mp
import numpy as np

#Mode debug
debugMode = False

#Vérification des args
if len(sys.argv) > 1:
    debugMode = True

# Initialiser MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialiser la capture vidéo.
cap = cv2.VideoCapture(0)

# Variables pour compter les squats.
squat_count = 0
down_position = False


def calculate_angle(a, b, c) -> int:
    """
    Calcule l'angle entre trois points en degrés.

    Retourne :
    angle -- L'angle en degrés formé par les segments b-a et b-c.

    Pour calculer l'angle entre les segments b-a et b-c, la fonction utilise
    la formule de l'arc tangente (arctan2) et convertit le résultat en degrés.
    L'angle résultant est toujours positif et mesure l'angle le plus petit entre
    les deux segments. Si l'angle dépasse 180 degrés, il est corrigé pour
    refléter l'angle correspondant entre 0 et 180 degrés.
    """
    # Convertir les points a, b, c en tableaux numpy pour faciliter les calculs.
    a = np.array(a)  # Premier point.
    b = np.array(b)  # Deuxième point.
    c = np.array(c)  # Troisième point.

    # Calculer l'angle en radians entre les points b-c et b-a.
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])

    # Convertir l'angle en degrés.
    angle = np.abs(radians * 180.0 / np.pi)

    # Si l'angle est supérieur à 180 degrés, corriger sa valeur à 360 degrés moins cet angle.
    if angle > 180.0:
        angle = 360 - angle

    # Retourner l'angle calculé.
    return angle


while cap.isOpened():
    ret, frame = cap.read()

    # Convertir l'image en RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Convertir l'image en BGR pour l'affichage.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extraire les landmarks.
    if results.pose_landmarks:

        # Récupérer les dimensions de la capture vidéo.
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        image_dims = [frame_width, frame_height]

        landmarks = results.pose_landmarks.landmark

        # Définir les points pour les articulations principales.
        keypoints = {
            'left_hip': [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
            'left_knee': [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
            'left_ankle': [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
            'right_hip': [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
            'right_knee': [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
            'right_ankle': [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y],
            'left_shoulder': [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            'left_elbow': [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
            'left_wrist': [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y],
            'right_shoulder': [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
            'right_elbow': [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
            'right_wrist': [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y],
        }

        # Convertir les coordonnées normalisées en pixels.
        keypoints_pixel = {k: np.multiply(v, image_dims).astype(int) for k, v in keypoints.items()}

        # Dessiner les points pour toutes les articulations.
        if debugMode:
            for point in keypoints_pixel.values():
                cv2.circle(image, tuple(point), 5, (0, 0, 255), -1)

        # Dessiner les lignes pour représenter le squelette.
        if debugMode:
            skeleton = [
                ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
                ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
                ('left_shoulder', 'right_shoulder'),
                ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
                ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
                ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
                ('left_hip', 'right_hip')
            ]

            for line in skeleton:
                cv2.line(image, tuple(keypoints_pixel[line[0]]), tuple(keypoints_pixel[line[1]]), (0, 255, 0), 2)

        # Calculer les angles des genoux.
        left_knee_angle = calculate_angle(keypoints['left_hip'], keypoints['left_knee'], keypoints['left_ankle'])
        right_knee_angle = calculate_angle(keypoints['right_hip'], keypoints['right_knee'], keypoints['right_ankle'])

        # Détecter la position basse.
        if left_knee_angle < 100 and right_knee_angle < 100 and not down_position:
            down_position = True

        # Détecter la position haute.
        if left_knee_angle > 160 and right_knee_angle > 160 and down_position:
            squat_count += 1
            down_position = False

        # Afficher le compteur de squats.
        cv2.putText(image, f'Squats: {squat_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Afficher les angles.
        if debugMode:
            cv2.putText(image, f'Left Angle: {int(left_knee_angle)}', tuple(keypoints_pixel['left_knee']),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Right Angle: {int(right_knee_angle)}', tuple(keypoints_pixel['right_knee']),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Afficher l'image.
    cv2.imshow('Squat Counter', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
