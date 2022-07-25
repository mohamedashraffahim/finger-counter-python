import cv2
import time
import os
import HandTrackingModule as htm

cam = cv2.VideoCapture(0)
folderPath = "D:\Mohamed Ashraf\college\Finger Python Project\Images"
imgList = os.listdir(folderPath)
# print(imgList)
overLayer = []
for img in imgList:
    image = cv2.imread(f'{folderPath}/{img}')
    overLayer.append(image)
# print(len(overLayer))

detector = htm.handDetector(detectionCon=1)

pTime = 0

tipIds = [4, 8, 12, 16, 20]

while True:
    show, frame = cam.read()

    frame = detector.findHands(frame)
    lmlist = detector.findPosition(frame, draw=False)
    # print(lmlist)

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
        # print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)
        h, w, c = overLayer[totalFingers].shape
        frame[0:h, 0:w] = overLayer[totalFingers]
        # cv2.rectangle(frame, (20, 225), (170, 425), (75, 0, 130), cv2.FILLED)
        # cv2.putText(frame, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
        #             10, (130, 0, 130), 25)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
