import cv2
import time
import HandTracking as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.hand_detection()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.detect_hands(img, draw=True)
    lmList = detector.detect_position(img, draw=False)

    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Real Time Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
