import cv2
import mediapipe as mp 
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions  = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode = VisionRunningMode.IMAGE,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 5), (5, 9), (9, 13), (13, 17), (17, 0),     # Wrist
    (0, 1), (1, 2), (2, 3), (3, 4),                 # Thumb
    (5, 6), (6, 7), (7, 8),                         # Index
    (9, 10), (10, 11), (11, 12),                    # Middle
    (13, 14), (14, 15), (15, 16),                   # Ring
    (17, 18), (18, 19), (19, 20)                    # Pinky
]


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            
            h, w, _ = img.shape
            lm_list = []
            
            for lm in hand:
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            for start, end in HAND_CONNECTIONS:
                cv2.line(img, lm_list[start], lm_list[end], (0, 255, 0), 2)
            
            for x, y in lm_list:
                cv2.circle(img, (x, y), 5, (0, 0, 255), cv2.FILLED)



    # cv2.resizeWindow(img, 1000,800)
    cv2.imshow("Youtube Control By Hand Gesture", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows