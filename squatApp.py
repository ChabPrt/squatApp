import sys
import cv2
import mediapipe as mp
import numpy as np
import time

# Mode debug
debugMode = False

# Vérification des args
if len(sys.argv) > 1:
    debugMode = True

# Initialiser MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialiser MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialiser la capture vidéo.
cap = cv2.VideoCapture(0)

# Variables pour compter les squats.
squat_count = 0
down_position = False

# Variables pour le décompte
hand_in_box = False
countdown_started = False
countdown_finished = False
countdown_time = 0

def calculate_angle(a, b, c) -> int:
    """
    Calcule l'angle entre trois points en degrés.

    Retourne :
    angle -- L'angle en degrés formé par les segments b-a et b-c.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

while cap.isOpened():
    ret, frame = cap.read()

    # Convertir l'image en RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(image)
    results_hands = hands.process(image)

    # Convertir l'image en BGR pour l'affichage.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not countdown_finished and not countdown_started:
        # Définir la zone pour la main.
        box_top_left = (int(frame_width * 0.75), 0)
        box_bottom_right = (frame_width, int(frame_height * 0.25))
        cv2.rectangle(image, box_top_left, box_bottom_right, (0, 255, 0), 2)

        # Afficher le message d'invite.
        cv2.putText(image, 'Mettez votre main dans le carre',
                    (frame_width // 8 - 40, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(image, 'vert pour lancer le compteur !',
                    (frame_width // 8 - 40 , frame_height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * frame_width), int(lm.y * frame_height)
                if box_top_left[0] <= cx <= box_bottom_right[0] and box_top_left[1] <= cy <= box_bottom_right[1]:
                    hand_in_box = True
                    break
            if hand_in_box:
                break
        else:
            hand_in_box = False

    if hand_in_box and not countdown_started:
        countdown_started = True
        countdown_time = time.time()

    if countdown_started and not countdown_finished:
        elapsed_time = time.time() - countdown_time
        if elapsed_time > 3:
            countdown_finished = True
        else:
            count_display = 3 - int(elapsed_time)
            cv2.putText(image, str(count_display), (frame_width // 2 - 50, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

    if countdown_finished:
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark

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

            keypoints_pixel = {k: np.multiply(v, [frame_width, frame_height]).astype(int) for k, v in keypoints.items()}

            if debugMode:
                for point in keypoints_pixel.values():
                    cv2.circle(image, tuple(point), 5, (0, 0, 255), -1)

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

            left_knee_angle = calculate_angle(keypoints['left_hip'], keypoints['left_knee'], keypoints['left_ankle'])
            right_knee_angle = calculate_angle(keypoints['right_hip'], keypoints['right_knee'], keypoints['right_ankle'])

            if left_knee_angle < 100 and right_knee_angle < 100 and not down_position:
                down_position = True

            if left_knee_angle > 160 and right_knee_angle > 160 and down_position:
                squat_count += 1
                down_position = False

            cv2.putText(image, f'Squats: {squat_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if debugMode:
                cv2.putText(image, f'Left Angle: {int(left_knee_angle)}', tuple(keypoints_pixel['left_knee']),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Right Angle: {int(right_knee_angle)}', tuple(keypoints_pixel['right_knee']),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Squat Counter', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()