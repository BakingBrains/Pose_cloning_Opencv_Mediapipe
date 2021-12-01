import cv2
import mediapipe as mp
import numpy as np

# take video input for pose detection
# you can put here video of your choice
# take live camera  input for pose detection
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# initialize mediapipe pose solution
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# read each frame
while True:
    ret, img = cap.read()

    # do Pose detection
    results = pose.process(img)

    # draw extracted pose on desired position (left)
    mp_draw.draw_landmarks(img[0:720, 0:720], results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 5, 2)
                           )

    # draw extracted pose on desired position (right)
    mp_draw.draw_landmarks(img[0:720, 720:1280], results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 255, 0), 5, 2)
                           )

    cv2.imshow("Extracted Pose", img)

    # print all landmarks
    print(results.pose_landmarks)
    cv2.waitKey(1)