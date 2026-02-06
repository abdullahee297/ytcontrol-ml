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
    num_hands = 2
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

pause_state = False
mute_state = False
mute_lock = False
speed_state = False
speed_lock = False
fullscreen_state = False
fullscreen_lock = False
space_held = False
state = ""


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)

    result = detector.detect(mp_image)

    finger_count = 0

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            
            h, w, _ = img.shape
            lm_list = []
            

            # Cordinates for the Circles
            for lm in hand:
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Lines joining the coordinates
            for start, end in HAND_CONNECTIONS:
                cv2.line(img, lm_list[start], lm_list[end], (0, 255, 0), 2)
            
            # Circles Drawn
            for x, y in lm_list:
                cv2.circle(img, (x, y), 5, (0, 0, 255), cv2.FILLED)

            

            if lm_list[4][0] > lm_list[3][0] and not pause_state:
                print("Open Hand / Pause the video")
                pause_state = True
                pyautogui.press("k")

            elif lm_list[4][0] < lm_list[3][0] and pause_state:
                print("Close Hand / Play the video")
                pause_state = False
                pyautogui.press("k")


            # Finger Count
            # 1) Mute 
            # 2) 2X speed
            # 3) Full screen


            if lm_list[4][0] > lm_list[3][0]:
                finger_count += 1

            # Other fingers (y-axis check)
            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]

            for tip, pip in zip(tips, pips):
                if lm_list[tip][1] < lm_list[pip][1]:
                    finger_count += 1

            choice = finger_count

            match choice:

                # 1 Finger → Mute / Unmute
                case 1:
                    if not mute_lock:
                        if not mute_state:
                            state = "Mute"
                            pyautogui.press("m")
                            mute_state = True
                        else:
                            state = "Unmute"
                            pyautogui.press("m")
                            mute_state = False
                        mute_lock = True

                # 2 Fingers - 2x Speed
                case 2:
                    if not speed_lock:
                        if not speed_state:
                            state = "2x Speed ON"
                            pyautogui.hotkey("shift", ">")
                            speed_state = True
                        else:
                            state = "Normal Speed"
                            pyautogui.hotkey("shift", "<")
                            speed_state = False
                        speed_lock = True

                # 3 Fingers Fullscreen ON / OFF
                case 3:
                    if not fullscreen_lock:
                        if not fullscreen_state:
                            state = "Fullscreen ON"
                            pyautogui.press("f")
                            fullscreen_state = True
                        else:
                            state = "Fullscreen OFF"
                            pyautogui.press("f")
                            fullscreen_state = False
                        fullscreen_lock = True

                # No valid gesture Reset locks
                case _:
                    mute_lock = False
                    speed_lock = False
                    fullscreen_lock = False
                    
            cv2.putText(img,
                        state,
                        (10,200),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255,0 ,0),
                        2
                    ) 
    
    if pause_state:
            cv2.putText(img,
            "Pause",
            (10,100),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255,0 ,0),
            2
            ) 

    else:
            cv2.putText(img,
            "Play",
            (10,100),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255,0 ,0),
            2
            )


    # cv2.resizeWindow(img, 1000,800)
    cv2.imshow("Youtube Control By Hand Gesture", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows