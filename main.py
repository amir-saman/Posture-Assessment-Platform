import cv2
import mediapipe as mp

# Create instances of mediapipe pose (for human pose estimation) and drawing utilities (for drawing on images)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create video feed from the default camera
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('Mediapipe Feed', frame)

    # If "x" is clicked then window is closed
    if cv2.waitKey(10) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
