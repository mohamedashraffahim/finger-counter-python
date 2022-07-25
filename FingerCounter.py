import cv2
import time
import os
import HandTrackingModule as htm
#Some devices you need to put 1 instead of 0 because of the camera number
cam = cv2.VideoCapture(0)
#Put here the path of the folder of finger Images
folderPath = "Images"
imgList = os.listdir(folderPath)

overLayer = []
for img in imgList:
    image = cv2.imread(f'{folderPath}/{img}')
    overLayer.append(image)
detector = htm.handDetector(detectionCon=1)
pTime = 0
tipIds = [4, 8, 12, 16, 20]

while True:
    show, frame = cam.read()
    frame = detector.findHands(frame)
    lmlist = detector.findPosition(frame, draw=False)
    if len(lmlist) != 0:
        fingers = []
        #For Thumb
        if lmlist[4][1] < lmlist[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #For 4 Fingers
        for i in range(1, 5):
            if lmlist[int(tipIds[i])][2] < lmlist[int(tipIds[i])-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)
        h, w, c = overLayer[totalFingers].shape
        frame[0:h, 0:w] = overLayer[totalFingers]
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("Frame", frame)
    #Press 'q' to close the program and you can change it to any letter you want
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
