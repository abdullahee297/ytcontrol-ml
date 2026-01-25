import cv2
import mediapipe as mp 
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        break



    cv2.imshow("Youtube Control By Hand Gesture", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows