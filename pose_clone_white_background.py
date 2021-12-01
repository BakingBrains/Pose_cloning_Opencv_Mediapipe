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

# read each frame/image from capture object
while True:
    ret, img = cap.read()

    # do Pose detection
    results = pose.process(img)

    # draw the detected pose on frame
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                           )
    # # Display the output
    cv2.imshow("Pose Estimation", img)

    # Extract and draw pose on plain white image
    h, w, c = img.shape   # get shape of original frame
    newImg = np.zeros([h, w, c])  # create blank image with original frame size
    newImg.fill(255)  # set white background. put 0 if you want to make it black

    # draw extracted pose on the new image
    mp_draw.draw_landmarks(newImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 5, 2)
                           )

    cv2.imshow("Extracted Pose", newImg)

    # print all landmarks
    print(results.pose_landmarks)
    cv2.waitKey(1)