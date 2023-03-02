import cv2
import time
import math as m
import mediapipe as mp



# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Calculate angle
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree


data = []
score = 0

# Function to send alert when bad posture detected
def sendWarning():
    pass


# Initilise bad frame counter
bad_frames = 0

# Font type
font = cv2.FONT_HERSHEY_PLAIN

# Colours
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Initialise mediapipe pose class
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

start = time.time()

while cap.isOpened():

    success, image = cap.read()
    if not success:
        print("Null.Frames")
        break
    # Get frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Get height and width
    h, w = image.shape[:2]

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image
    keypoints = pose.process(image)

    # Convert the image back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # lm (LandMark) and lmPose to represent the mediapipe pose landmarks
    lm = keypoints.pose_landmarks
    lmPose = mp_pose.PoseLandmark

    # Get the landmark coordinates
    # Left shoulder
    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
    # Right shoulder
    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
    # Left ear
    l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
    l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
    # Left hip
    l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
    l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

    # Calculate distance between left shoulder and right shoulder points
    offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

    # Align the camera to point at the side view of the person
    if offset < 75:
        cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
    else:
        cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

    # Calculate angles
    neck_angle = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
    torso_angle = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

    # Draw landmarks
    cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
    cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)

    cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
    cv2.circle(image, (r_shldr_x, r_shldr_y), 7, blue, -1)
    cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

    cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

    # Draw text, Posture and angle inclination
    # Text string for display
    angle_text_string = 'Neck : ' + str(int(neck_angle)) + '  Torso : ' + str(int(torso_angle))

    # Decide if the posture is good or bad
    if neck_angle < 40 and torso_angle < 10:
        data.append((round((time.time() - start), 2), score + 1))
        bad_frames = 0

        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
        cv2.putText(image, str(int(neck_angle)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
        cv2.putText(image, str(int(torso_angle)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)

        # Join landmarks.
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)

# extract joint angles
    else:
        data.append((round((time.time() - start), 2), score - 1))

        bad_frames += 1

        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
        cv2.putText(image, str(int(neck_angle)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
        cv2.putText(image, str(int(torso_angle)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

        # Join landmarks.
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)


    # Calculate the duration of posture
    bad_time = (1 / fps) * bad_frames

    # When posture is bad for a certain amount of time then a warning is sent
    if bad_time > 3:
        sendWarning()


    cv2.imshow('Posture Assessment', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        end = time.time()
        elapsed_time = end - start
        print("the session was " + str(elapsed_time) + " seconds long")
        print(data)

        break

cap.release()
cv2.destroyAllWindows()
