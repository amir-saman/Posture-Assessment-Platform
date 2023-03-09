import cv2
import time
import math as m
import mediapipe as mp
from notify_run import Notify
import pandas as pd



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


# Initialise Notify
notify = Notify()
run_once = 0

sleep_time = 1

total_neck_and_torso_score = 0
total_torso_score = 0
total_neck_score = 0

current_neck_score_data = [] # = [time, neck angle, RULA Score]
current_torso_score_data = []
# cumulative_neck_torso_score_data = []
current_neck_torso_score_data = [] #time, neck score, torso score, Overall according to RULA
cumulative_neck_score_data = [] # = [time, cumulative RULA score]
cumulative_torso_score_data = []

# Initilise frame counters
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
    if offset <= 120 and offset > 110:
        cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
    else:
        cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

    # Calculate angles
    neck_angle = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
    torso_angle = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

    # Draw landmarks
    cv2.circle(image, (l_shldr_x, l_shldr_y), 7, pink, -1)
    cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)

    cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
    cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
    cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

    cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

    # Text string for display
    angle_text_string = 'Neck : ' + str(int(neck_angle)) + '  Torso : ' + str(int(torso_angle))



    # Assess posture and score it based on RULA
    if torso_angle == 0:
        current_torso_score_data.append((round((time.time() - start), 2), torso_angle, 1))

        total_torso_score = total_torso_score + 1

        cumulative_torso_score_data.append((round((time.time() - start), 2), total_torso_score))

        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, blue, 2)
        cv2.putText(image, str(int(neck_angle)), (l_shldr_x + 10, l_shldr_y), font, 0.9, blue, 2)
        cv2.putText(image, str(int(torso_angle)), (l_hip_x + 10, l_hip_y), font, 0.9, blue, 2)

        # Join landmarks.
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), blue, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), blue, 4)

        time.sleep(sleep_time)


    if torso_angle < 20 and torso_angle != 0:
        current_torso_score_data.append((round((time.time() - start), 2), torso_angle, 2))

        total_torso_score = total_torso_score + 2

        cumulative_torso_score_data.append((round((time.time() - start), 2), total_torso_score))

        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, green, 2)
        cv2.putText(image, str(int(neck_angle)), (l_shldr_x + 10, l_shldr_y), font, 0.9, green, 2)
        cv2.putText(image, str(int(torso_angle)), (l_hip_x + 10, l_hip_y), font, 0.9, green, 2)

        # Join landmarks.
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)

        time.sleep(sleep_time)


    if torso_angle < 60 and torso_angle >= 20:
        current_torso_score_data.append((round((time.time() - start), 2), torso_angle, 3))

        total_torso_score = total_torso_score + 3

        cumulative_torso_score_data.append((round((time.time() - start), 2), total_torso_score))

        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, yellow, 2)
        cv2.putText(image, str(int(neck_angle)), (l_shldr_x + 10, l_shldr_y), font, 0.9, yellow, 2)
        cv2.putText(image, str(int(torso_angle)), (l_hip_x + 10, l_hip_y), font, 0.9, yellow, 2)

        # Join landmarks.
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), yellow, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), yellow, 4)

        time.sleep(sleep_time)


    if torso_angle >= 60:
        current_torso_score_data.append((round((time.time() - start), 2), torso_angle, 4))

        total_torso_score = total_torso_score + 4

        cumulative_torso_score_data.append((round((time.time() - start), 2), total_torso_score))

        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
        cv2.putText(image, str(int(neck_angle)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
        cv2.putText(image, str(int(torso_angle)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

        # Join landmarks.
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

        #time.sleep(sleep_time)




    #decide if the neck posture is good or bad

    if neck_angle < 10 and neck_angle >= 0:
        current_neck_score_data.append((round((time.time() - start), 2), torso_angle, 1))

        total_neck_score = total_neck_score + 1

        cumulative_neck_score_data.append((round((time.time() - start), 2), total_neck_score))

        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, green, 2)
        cv2.putText(image, str(int(neck_angle)), (l_shldr_x + 10, l_shldr_y), font, 0.9, green, 2)
        cv2.putText(image, str(int(torso_angle)), (l_hip_x + 10, l_hip_y), font, 0.9, green, 2)

        # Join landmarks.
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)


    if neck_angle < 20 and neck_angle >=10:
        current_neck_score_data.append((round((time.time() - start), 2), torso_angle, 2))

        total_neck_score = total_neck_score + 2

        cumulative_neck_score_data.append((round((time.time() - start), 2), total_neck_score))

        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, yellow, 2)
        cv2.putText(image, str(int(neck_angle)), (l_shldr_x + 10, l_shldr_y), font, 0.9, yellow, 2)
        cv2.putText(image, str(int(torso_angle)), (l_hip_x + 10, l_hip_y), font, 0.9, yellow, 2)

        # Join landmarks.
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), yellow, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), yellow, 4)


    if neck_angle >= 20:
        current_neck_score_data.append((round((time.time() - start), 2), torso_angle, 3))

        total_neck_score = total_neck_score + 3

        cumulative_neck_score_data.append((round((time.time() - start), 2), total_neck_score))

        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
        cv2.putText(image, str(int(neck_angle)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
        cv2.putText(image, str(int(torso_angle)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

        # Join landmarks.
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)

    # Calculate the duration of posture
    bad_time = (1 / fps) * bad_frames


    # If you stay in bad posture for more than x seconds send warning

    if bad_time > 3:
        if run_once == 0:
            notify.send('test')
            run_once = 1
            bad_time = 0
    else:
        run_once = 0

    cv2.imshow('Posture Assessment', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        end = time.time()
        elapsed_time = end - start
        print("the session was " + str(elapsed_time) + " seconds long")
        print(total_torso_score)

        for i in range(len(current_neck_score_data)):

            if current_neck_score_data[i][2] == 1:
                match current_torso_score_data[i][2]:
                    case 1:
                        current_neck_torso_score_data.append((current_neck_score_data[i][0], 1, 1, 1))
                    case 2:
                        current_neck_torso_score_data.append((current_neck_score_data[i][0], 1, 2, 2))
                    case 3:
                        current_neck_torso_score_data.append((current_neck_score_data[i][0], 1, 3, 3))
                    case 4:
                        current_neck_torso_score_data.append((current_neck_score_data[i][0], 1, 4, 5))

            if current_neck_score_data[i][2] == 2:
                match current_torso_score_data[i][2]:
                    case 1:
                        current_neck_torso_score_data.append((current_neck_score_data[i][0], 2, 1, 2))
                    case 2:
                        current_neck_torso_score_data.append((current_neck_score_data[i][0], 2, 2, 2))
                    case 3:
                        current_neck_torso_score_data.append((current_neck_score_data[i][0], 2, 3, 4))
                    case 4:
                        current_neck_torso_score_data.append((current_neck_score_data[i][0], 2, 4, 5))

            if current_neck_score_data[i][2] == 3:
                match current_torso_score_data[i][2]:
                    case 1:
                        current_neck_torso_score_data.append((current_neck_score_data[i][0], 3, 1, 3))
                    case 2:
                        current_neck_torso_score_data.append((current_neck_score_data[i][0], 3, 2, 3))
                    case 3:
                        current_neck_torso_score_data.append((current_neck_score_data[i][0], 3, 3, 4))
                    case 4:
                        current_neck_torso_score_data.append((current_neck_score_data[i][0], 3, 4, 5))


        # Convert the list of tuples to a pandas DataFrame
        cumulative_torso_score_data_plot = pd.DataFrame(cumulative_torso_score_data, columns=['Time Elapsed', 'Cumulative Torso Score'])
        current_torso_score_data_plot = pd.DataFrame(current_torso_score_data, columns=['Time Elapsed', 'Torso Angle', 'Torso Score'])

        cumulative_neck_score_data_plot = pd.DataFrame(cumulative_neck_score_data, columns=['Time Elapsed', 'Cumulative Neck Score'])
        current_neck_score_data_plot = pd.DataFrame(current_neck_score_data, columns=['Time Elapsed', 'Neck Angle', 'Neck Score'])

        # cumulative_neck_torso_score_data_plot = pd.DataFrame(cumulative_neck_torso_score_data, columns=['Time Elapsed', 'Cumulative Neck & Torso Score'])

        current_neck_torso_score_data_plot = pd.DataFrame(current_neck_torso_score_data, columns=['Time Elapsed', 'Neck Score', 'Torso Score', 'Combined RULA Score'])



        # Save the DataFrame to an excel file
        cumulative_torso_score_data_plot.to_excel('Cumulative Torso Score.xlsx', index=False)
        current_torso_score_data_plot.to_excel('Torso Score at Any Given Point.xlsx', index=False)

        cumulative_neck_score_data_plot.to_excel('Cumulative Neck Score.xlsx', index=False)
        current_neck_score_data_plot.to_excel('Neck Score at Any Given Point.xlsx', index=False)

        #cumulative_neck_torso_score_data_plot.to_excel('Cumulative Neck & Torso Score.xlsx', index=False)

        current_neck_torso_score_data_plot.to_excel('Current Neck & Torso Combined Score.xlsx', index=False)


        break

cap.release()
cv2.destroyAllWindows()
